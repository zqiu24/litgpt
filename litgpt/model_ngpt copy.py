# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# The text below is the original header from the nanoGPT library
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from torch.optim.optimizer import T

from litgpt.config import Config


def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    # Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # Apply the rotary embeddings to the query and key
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)
    return q_rot, k_rot

def get_sinusoidal_embeddings( n_positions, dim):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim))
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb


class Block(nn.Module):

    def __init__(self, config, iblock):
        super().__init__()
        self.config = config

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        
        self.c_fc    = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.silu    = nn.SiLU()
        self.mlp_c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)

        if (config.use_nGPT == 0):
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)

        if (config.use_nGPT == 1):
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.sqk_init_value = 1.0       
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(self.suv_init_scaling*torch.ones(2 * 4 * config.n_embd, dtype=torch.float32))

    
    def justnorm(self, x):
        #return F.normalize(x, p=2, dim=-1)
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, h):
        B, T, C = h.size()

        hin = h
        if (self.config.use_nGPT == 0):
            hin = self.rmsnorm_att(h)
        
        q = self.query(hin)
        k = self.key(hin)
        v = self.value(hin)

        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head) 
        k = k.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)

        sinusoidal_pos = get_sinusoidal_embeddings(T, self.config.n_embd // self.config.n_head).to(device=q.device)
        q, k = apply_rotary_position_embeddings(sinusoidal_pos, q.transpose(1, 2), k.transpose(1, 2))
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)

        if (self.config.use_nGPT == 1):
            sqk = (self.sqk * (self.sqk_init_value/self.sqk_init_scaling)).view(1, 1, self.config.n_head, self.config.n_embd // self.config.n_head)
            q = sqk * self.justnorm(q)  
            k = sqk * self.justnorm(k)  

        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5
        if (self.config.use_nGPT == 0): softmax_scale = 1.0 / sqrt_head_dim 
        if (self.config.use_nGPT == 1): softmax_scale = sqrt_head_dim 
        y = flash_attn_func(q.to(dtype=torch.bfloat16), k.to(dtype=torch.bfloat16), v.to(dtype=torch.bfloat16), dropout_p=0.0, softmax_scale=softmax_scale, causal=True, window_size=(-1, -1), alibi_slopes=None, deterministic=True)
        y = y.to(dtype=q.dtype)
        y = y.contiguous().view(B, T, self.config.n_embd)

        h_att = self.att_c_proj(y)

        if (self.config.use_nGPT == 0):
            h = h + h_att
        if (self.config.use_nGPT == 1):
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)
            
            A_norm = self.justnorm(h) # normally, normalization is not needed
            B_norm = self.justnorm(h_att)
                
            #res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        hin = h
        if (self.config.use_nGPT == 0):
            hin = self.rmsnorm_mlp(h)
        uv = self.c_fc(hin)
        if (self.config.use_nGPT == 1):
            suv = (self.suv * ((self.suv_init_value/self.suv_init_scaling) * (self.config.n_embd ** 0.5))) 
            uv = suv * uv  
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        if (self.config.use_nGPT == 0):
            h = h + h_mlp
        if (self.config.use_nGPT == 1):
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = self.justnorm(h) # normally, normalization is not needed
            B_norm = self.justnorm(h_mlp)
                
            #res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        return h

'''
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0 ** 0.5)    # 1 / sqrt(n_embd)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = False 
'''

class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.padded_vocab_size is not None
        self.config = config

        config.base_scale = 1.0 / (config.n_embd ** 0.5)
        config.use_nGPT = 1
        config.dropout = 0.0

        self.transformer = nn.ModuleDict(
            dict(
                # wte = nn.Embedding(config.vocab_size, config.n_embd),
                wte = nn.Embedding(config.padded_vocab_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                # h = nn.ModuleList([Block(config, block_idx) for block_idx in range(config.n_layer)])
                h = nn.ModuleList(Block(config, block_idx) for block_idx in range(config.n_layer))
            )
        )
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # *we don't use it becuase in the nGPT paper there was no weight tying of weights*
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        # self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=config.base_scale/math.sqrt(2 * config.n_layer))
        # report number of parameters
        # print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
        if (config.use_nGPT == 1):
            self.sz_init_value = 1.00
            self.sz_init_scaling = config.base_scale
            # self.sz = torch.nn.Parameter(self.sz_init_scaling*torch.ones(config.vocab_size, dtype=torch.float32))
            self.sz = torch.nn.Parameter(self.sz_init_scaling*torch.ones(config.padded_vocab_size, dtype=torch.float32))

        if (config.use_nGPT == 0):
            self.rmsnorm_f = RMSNorm(config.n_embd)

        self.max_seq_length = self.config.block_size


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        if self.max_seq_length < t:
            raise ValueError(f"Cannot forward sequence of length {t}, max seq length is only {self.max_seq_length}.")
        
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block_idx, block in enumerate(self.transformer.h):
            x = block(x)

        if (self.config.use_nGPT == 0):
            x = self.rmsnorm_f(x)

        logits = self.lm_head(x)
        if (self.config.use_nGPT == 1):
            sz = self.sz * (self.sz_init_value/self.sz_init_scaling)
            logits = sz * logits

        '''
        if targets is not None:
            logits = self.lm_head(x)
            if (self.config.use_nGPT == 1):
                sz = self.sz * (self.sz_init_value/self.sz_init_scaling)
                logits = sz * logits
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            if (self.config.use_nGPT == 1):
                sz = self.sz * (self.sz_init_value/self.sz_init_scaling)
                logits = sz * logits
            # loss = None
        '''

        return logits # , loss


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = False#fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

