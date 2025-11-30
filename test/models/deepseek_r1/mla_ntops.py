"""
DeepSeek-R1 MLA (Multi-Head Latent Attention) 基于 ntops 的实现
使用 kv_cache + pe_cache 缓存策略（absorb 模式）
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试动态添加 ntops 源码路径，避免未安装时报错
_REPO_ROOT = Path(__file__).resolve().parents[5]
_NTOPS_SRC = _REPO_ROOT / "InfiniCore" / "ntops" / "src"
if _NTOPS_SRC.exists() and str(_NTOPS_SRC) not in sys.path:
    sys.path.insert(0, str(_NTOPS_SRC))

try:
    import ntops.torch as ntops_torch  # type: ignore

    NTOPS_AVAILABLE = True
except Exception:  # pragma: no cover - 构建环境缺失时使用 PyTorch fallback
    ntops_torch = None
    NTOPS_AVAILABLE = False


@dataclass
class DeepSeekR1MLAConfig:
    """DeepSeek-R1 MLA 关键参数（来自官方 config.json）。"""

    hidden_size: int = 7168
    max_seq_len: int = 4096 * 4
    max_batch_size: int = 16
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0
    original_max_position_embeddings: int = 4096
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    rms_norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    """优先调用 ntops.rms_norm 的 RMSNorm。"""

    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if NTOPS_AVAILABLE:
            return ntops_torch.rms_norm(x, (self.dim,), weight=self.weight, eps=self.eps)
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(config: DeepSeekR1MLAConfig, device: str = "cuda") -> torch.Tensor:
    """按照 YaRN 扩展规则预计算 RoPE 复数频率表。"""

    dim = config.qk_rope_head_dim
    seqlen = config.max_seq_len
    base = config.rope_theta
    factor = config.rope_factor

    def find_correction_dim(num_rotations: float) -> float:
        return dim * math.log(
            config.original_max_position_embeddings / (num_rotations * 2 * math.pi)
        ) / (2 * math.log(base))

    half_dim = dim // 2

    def linear_ramp_factor(low: float, high: float) -> torch.Tensor:
        if low == high:
            high += 0.001
        linear = (
            torch.arange(half_dim, dtype=torch.float32, device=device) - low
        ) / (high - low)
        return torch.clamp(linear, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if seqlen > config.original_max_position_embeddings:
        low = max(0, math.floor(find_correction_dim(config.beta_fast)))
        high = min(half_dim - 1, math.ceil(find_correction_dim(config.beta_slow)))
        smooth = 1 - linear_ramp_factor(low, high)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """对输入张量应用 RoPE。"""

    dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    y = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return y.to(dtype)


def mla_attention_impl(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    wkv_b_nope: torch.Tensor,
    wkv_b_value: torch.Tensor,
    mask: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """调用 ntops 或 PyTorch fallback 执行注意力。"""

    if NTOPS_AVAILABLE:
        return ntops_torch.mla_attention(
            q_nope,
            q_pe,
            kv_cache,
            pe_cache,
            wkv_b_nope,
            wkv_b_value,
            mask,
            scale,
        )

    q_compressed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_nope)
    scores = (
        torch.einsum("bshc,btc->bsht", q_compressed, kv_cache)
        + torch.einsum("bshr,btr->bsht", q_pe, pe_cache)
    ) * scale
    if mask is not None:
        scores = scores + mask.unsqueeze(1)
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q_nope)
    weighted_kv = torch.einsum("bsht,btc->bshc", attn, kv_cache)
    # wkv_b_value: [heads, kv_lora_rank, v_dim]
    return torch.einsum("bshc,hcd->bshd", weighted_kv, wkv_b_value)


class MLANtops(nn.Module):
    """基于 ntops 的 MLA 层实现。"""

    def __init__(self, config: DeepSeekR1MLAConfig, dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        if self.q_lora_rank > 0:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=dtype)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps, dtype=dtype)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False, dtype=dtype)
        else:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False, dtype=dtype)

        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, dtype=dtype)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps, dtype=dtype)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            dtype=dtype,
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False, dtype=dtype)

        self.softmax_scale = self.qk_head_dim ** -0.5
        if config.max_seq_len > config.original_max_position_embeddings:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale *= mscale * mscale

        self.register_buffer(
            "kv_cache",
            torch.zeros(config.max_batch_size, config.max_seq_len, self.kv_lora_rank, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "pe_cache",
            torch.zeros(config.max_batch_size, config.max_seq_len, self.qk_rope_head_dim, dtype=dtype),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        if self.q_lora_rank > 0:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        else:
            q = self.wq(x)
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        kv_compressed = self.kv_norm(kv)
        self.kv_cache[:bsz, start_pos:end_pos] = kv_compressed
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        wkv_b = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)
        wkv_b_nope = wkv_b[:, : self.qk_nope_head_dim, :]
        # wkv_b_value 需要形状 [heads, kv_lora_rank, v_dim]
        wkv_b_value = wkv_b[:, -self.v_head_dim :, :].transpose(1, 2).contiguous()

        output = mla_attention_impl(
            q_nope,
            q_pe,
            self.kv_cache[:bsz, :end_pos],
            self.pe_cache[:bsz, :end_pos],
            wkv_b_nope,
            wkv_b_value,
            mask,
            self.softmax_scale,
        )

        return self.wo(output.flatten(2))

    def reset_cache(self) -> None:
        self.kv_cache.zero_()
        self.pe_cache.zero_()


def create_mla_ntops(device: str = "cuda", dtype=torch.bfloat16):
    """创建基于 ntops 的 MLA 模块并返回 (model, freqs, config)。"""

    config = DeepSeekR1MLAConfig()
    model = MLANtops(config, dtype=dtype).to(device=device)
    for param in model.parameters():
        if param.dtype == dtype:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
    freqs_cis = precompute_freqs_cis(config, device)
    return model, freqs_cis, config
