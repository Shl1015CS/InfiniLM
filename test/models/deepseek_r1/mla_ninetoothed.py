"""
MLA (Multi-Head Latent Attention) implementation using Ninetoothed operators
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass

# Try to import ninetoothed operators
try:
    from ntops import torch as nt_torch
    # Import MLA specific operators
    from ntops.torch.mla import mla_attention, mla_update_cache
    # Temporarily disable ninetoothed due to bfloat16 compilation issues
    # Will be enabled after resolving dtype compatibility
    NINETOOTHED_AVAILABLE = False
    print("Info: Ninetoothed MLA operators available at: ntops.torch.mla")
    print("Info: Temporarily using PyTorch fallback for bfloat16 compatibility")
except ImportError as e:
    print(f"Warning: Ninetoothed not available ({e}), falling back to PyTorch")
    NINETOOTHED_AVAILABLE = False


@dataclass
class MLAConfig:
    """MLA configuration parameters"""
    dim: int = 2048
    n_heads: int = 16
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    max_batch_size: int = 8
    max_seq_len: int = 16384
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


def precompute_freqs_cis(config: MLAConfig, device):
    """Precompute rotary position encoding frequencies"""
    dim = config.qk_rope_head_dim
    seqlen = config.max_seq_len
    base = config.rope_theta
    factor = config.rope_factor
    beta_fast = config.beta_fast
    beta_slow = config.beta_slow
    
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
    
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)
    
    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > config.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, config.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to(device)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position encoding"""
    if NINETOOTHED_AVAILABLE:
        # Use ninetoothed RoPE if available
        # Convert freqs_cis to sin/cos
        dtype = x.dtype
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis_view = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
        y = torch.view_as_real(x_complex * freqs_cis_view).flatten(3)
        return y.to(dtype)
    else:
        # Fallback to PyTorch
        dtype = x.dtype
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis_view = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
        y = torch.view_as_real(x_complex * freqs_cis_view).flatten(3)
        return y.to(dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization using Ninetoothed"""
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
    
    def forward(self, x: torch.Tensor):
        if NINETOOTHED_AVAILABLE:
            # Use ninetoothed RMS norm
            # Interface: rms_norm(input, normalized_shape, weight, eps)
            return nt_torch.rms_norm(x, (self.dim,), self.weight, self.eps)
        else:
            # Fallback to PyTorch
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class Linear(nn.Module):
    """Linear layer"""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype or torch.bfloat16))
        else:
            self.bias = None
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor):
        # For now, use PyTorch F.linear for compatibility
        # Ninetoothed mm can be optimized later
        return F.linear(x, self.weight, self.bias)


class MLANinetoothed(nn.Module):
    """
    Multi-Head Latent Attention using Ninetoothed operators
    Uses kv_cache + pe_cache caching strategy (absorb mode)
    """
    def __init__(self, config: MLAConfig, dtype=torch.bfloat16):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        # Q projection
        if self.q_lora_rank == 0:
            self.wq = Linear(self.dim, self.n_heads * self.qk_head_dim, dtype=dtype)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank, dtype=dtype)
            self.q_norm = RMSNorm(self.q_lora_rank, dtype=dtype)
            self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, dtype=dtype)
        
        # KV projection
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, dtype=dtype)
        self.kv_norm = RMSNorm(self.kv_lora_rank, dtype=dtype)
        self.wkv_b = Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), dtype=dtype)
        
        # Output projection
        self.wo = Linear(self.n_heads * self.v_head_dim, self.dim, dtype=dtype)
        
        # Softmax scaling factor
        self.softmax_scale = self.qk_head_dim ** -0.5
        if config.max_seq_len > config.original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        
        # Absorb mode cache: kv_cache + pe_cache
        self.register_buffer(
            "kv_cache",
            torch.zeros(config.max_batch_size, config.max_seq_len, self.kv_lora_rank, dtype=dtype),
            persistent=False
        )
        self.register_buffer(
            "pe_cache",
            torch.zeros(config.max_batch_size, config.max_seq_len, self.qk_rope_head_dim, dtype=dtype),
            persistent=False
        )
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with Ninetoothed operators
        
        Args:
            x: Input tensor (batch_size, seq_len, dim)
            start_pos: Sequence start position
            freqs_cis: Precomputed rotary encodings
            mask: Attention mask
        
        Returns:
            Output tensor (batch_size, seq_len, dim)
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Q projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Apply rotary encoding to Q
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # KV projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Apply rotary encoding to K
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # Absorb mode: use kv_cache + pe_cache
        wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)
        
        # Compress q_nope: [bsz, seqlen, n_heads, qk_nope_head_dim] @ [n_heads, qk_nope_head_dim, kv_lora_rank]
        # -> [bsz, seqlen, n_heads, kv_lora_rank]
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        
        # Update cache
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        
        # Compute attention scores
        # scores = q_nope @ kv_cache.T + q_pe @ pe_cache.T
        scores = (
            torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
            torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
        ) * self.softmax_scale
        
        if mask is not None:
            scores += mask.unsqueeze(1)
        
        # Softmax
        if NINETOOTHED_AVAILABLE:
            scores = nt_torch.softmax(scores, dim=-1)
        else:
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # Compute output
        # output = scores @ kv_cache @ wkv_b_value
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        
        return x
    
    def reset_cache(self):
        """Reset KV cache"""
        self.kv_cache.zero_()
        self.pe_cache.zero_()
