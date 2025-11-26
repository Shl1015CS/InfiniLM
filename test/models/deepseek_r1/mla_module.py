"""
DeepSeek-R1 MLA (Multi-Head Latent Attention) 自定义实现
使用 kv_cache + pe_cache 缓存策略
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLAConfig:
    """MLA 模块配置"""
    # 基础配置
    dim: int = 2048  # hidden_size
    max_seq_len: int = 4096 * 4
    max_batch_size: int = 8
    
    # Attention 头配置
    n_heads: int = 16  # num_attention_heads
    n_kv_heads: int = 16  # num_key_value_heads (对于 MLA，通常与 n_heads 相同)
    
    # MLA 特定配置
    q_lora_rank: int = 0  # Query LoRA rank (0 表示不使用 LoRA)
    kv_lora_rank: int = 512  # KV LoRA rank
    qk_nope_head_dim: int = 128  # Query/Key 非位置编码维度
    qk_rope_head_dim: int = 64  # Query/Key RoPE 维度
    v_head_dim: int = 128  # Value 头维度
    
    # RoPE 配置
    rope_theta: float = 10000.0
    original_seq_len: int = 4096
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    
    # 其他
    eps: float = 1e-6  # RMSNorm epsilon


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(config: MLAConfig, device: str = "cuda") -> torch.Tensor:
    """
    预计算 RoPE 的复数频率
    
    Args:
        config: MLA 配置
        device: 设备
    
    Returns:
        freqs_cis: 复数频率张量 [max_seq_len, qk_rope_head_dim // 2]
    """
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
        linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    
    # 计算基础频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    
    # YaRN 扩展（如果序列长度超过原始长度）
    if seqlen > config.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, config.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    
    # 计算位置编码
    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置编码
    
    Args:
        x: 输入张量 [..., seq_len, n_heads, head_dim]
        freqs_cis: 复数频率 [seq_len, head_dim // 2]
    
    Returns:
        应用 RoPE 后的张量
    """
    dtype = x.dtype
    # 转换为复数
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # 广播 freqs_cis 到正确的形状
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # 应用旋转
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) 模块
    使用 kv_cache + pe_cache 缓存策略（absorb 模式）
    """
    def __init__(self, config: MLAConfig, dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        # Q 投影（支持 LoRA）
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False, dtype=dtype)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=dtype)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=config.eps, dtype=dtype)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False, dtype=dtype)
        
        # KV 投影（LoRA 分解）
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, dtype=dtype)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.eps, dtype=dtype)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False, dtype=dtype)
        
        # 输出投影
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False, dtype=dtype)
        
        # Attention scale
        self.softmax_scale = self.qk_head_dim ** -0.5
        if config.max_seq_len > config.original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        
        # 使用 kv_cache + pe_cache 缓存策略
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
            x: 输入张量 [batch_size, seq_len, dim]
            start_pos: 当前序列的起始位置（用于缓存）
            freqs_cis: RoPE 频率 [seq_len, qk_rope_head_dim // 2]
            mask: 注意力掩码 [seq_len, seq_len]（可选）
        
        Returns:
            输出张量 [batch_size, seq_len, dim]
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Q 投影
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        
        # 分离 Q 的 nope 和 rope 部分
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # 对 Q 的 rope 部分应用旋转编码
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # KV 投影的第一阶段（压缩）
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # 对 K 的 rope 部分应用旋转编码
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # 使用 absorb 模式：kv_cache + pe_cache
        # 保存压缩后的 KV 和位置编码
        kv_compressed = self.kv_norm(kv)
        self.kv_cache[:bsz, start_pos:end_pos] = kv_compressed
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        
        # 计算注意力分数
        # Q_nope 需要与 wkv_b 的权重相乘来匹配压缩的 KV 空间
        wkv_b_weight = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)
        q_nope_compressed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_weight[:, :self.qk_nope_head_dim])
        
        # 计算分数：nope 部分 + rope 部分
        scores = (
            torch.einsum("bshc,btc->bsht", q_nope_compressed, self.kv_cache[:bsz, :end_pos]) +
            torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
        ) * self.softmax_scale
        
        # 应用掩码
        # mask: [seq_len, end_pos] -> [1, seq_len, 1, end_pos]
        # scores: [bsz, seq_len, n_heads, end_pos]
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(2)
        
        # Softmax
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # 加权聚合 V
        # 先聚合压缩的 KV，再通过 wkv_b 投影到 V 空间
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b_weight[:, -self.v_head_dim:])
        
        # 输出投影
        x = self.wo(x.flatten(2))
        
        return x
    
    def reset_cache(self):
        """重置缓存"""
        self.kv_cache.zero_()
        self.pe_cache.zero_()
