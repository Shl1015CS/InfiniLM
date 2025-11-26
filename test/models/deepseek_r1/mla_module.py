"""
DeepSeek-R1 MLA (Multi-Head Latent Attention) Implementation
Based on: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class MLAConfig:
    """MLA 配置参数"""
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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class Linear(nn.Module):
    """自定义线性层"""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def precompute_freqs_cis(config: MLAConfig, device) -> torch.Tensor:
    """预计算旋转位置编码的复数指数值"""
    dim = config.qk_rope_head_dim
    seqlen = config.max_seq_len
    beta_fast = config.beta_fast
    beta_slow = config.beta_slow
    base = config.rope_theta
    factor = config.rope_factor

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
    """应用旋转位置编码"""
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) 层
    使用 kv_cache + pe_cache 的缓存策略（absorb 模式）
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

        # Q 投影
        if self.q_lora_rank == 0:
            self.wq = Linear(self.dim, self.n_heads * self.qk_head_dim, dtype=dtype)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank, dtype=dtype)
            self.q_norm = RMSNorm(self.q_lora_rank, dtype=dtype)
            self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, dtype=dtype)
        
        # KV 投影
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, dtype=dtype)
        self.kv_norm = RMSNorm(self.kv_lora_rank, dtype=dtype)
        self.wkv_b = Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), dtype=dtype)
        
        # 输出投影
        self.wo = Linear(self.n_heads * self.v_head_dim, self.dim, dtype=dtype)
        
        # Softmax 缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5
        if config.max_seq_len > config.original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # 使用 absorb 模式的缓存：kv_cache + pe_cache
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
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, dim)
            start_pos: 序列起始位置
            freqs_cis: 预计算的旋转编码
            mask: 注意力掩码
        
        Returns:
            输出张量 (batch_size, seq_len, dim)
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Q 投影
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # 应用旋转编码到 Q
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # KV 投影
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # 应用旋转编码到 K
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # Absorb 模式：使用 kv_cache + pe_cache
        wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        
        # 更新缓存
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        
        # 计算注意力分数
        scores = (
            torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
            torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
        ) * self.softmax_scale
        
        if mask is not None:
            scores += mask.unsqueeze(1)
        
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # 计算输出
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        
        return x
    
    def reset_cache(self):
        """重置缓存"""
        self.kv_cache.zero_()
        self.pe_cache.zero_()
