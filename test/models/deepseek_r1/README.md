# DeepSeek-R1 MLA 测试

## 概述

本目录包含 DeepSeek-R1 MLA (Multi-Head Latent Attention) 模块的测试实现，包括：
- **自定义实现** (`mla_module.py`): 使用 `kv_cache + pe_cache` 缓存策略的 MLA 实现
- **测试脚本** (`attention_test.py`): 性能测试和正确性验证

## 文件说明

### mla_module.py
自定义 MLA 实现，核心特性：
- 使用 **absorb 模式**：`kv_cache + pe_cache` 缓存策略
- 支持 LoRA 低秩分解（Q 和 KV 投影）
- 支持 YaRN 扩展的 RoPE 位置编码
- BF16 精度

### attention_test.py
测试脚本，包含：
- 自定义实现的性能测试
- transformers 实现的性能测试（对比）
- 正确性验证

## 测试场景

### 1. 小批量预填充 (Prefill)
- **4个请求**，长度分别为 `[64, 128, 256, 256]`
- **历史长度**分别为 `[512, 0, 0, 256]`
- 输入和 KV Cache 随机初始化
- 推理 **100 轮**，计算平均延迟（ms per batch）

### 2. 大批量解码 (Decode)
- **16个请求**，输入长度均为 `1`
- **历史长度**为 `50×4, 100×4, 200×4, 400×4`
- 输入和 KV Cache 随机初始化
- 顺序推理 **100 轮**（Cache 会增长）
- 计算平均吞吐量（tokens/s）

## 使用方法

### 基础用法（仅测试自定义实现）

```bash
cd /data/users/shihaolin/infini/InfiniLM/test/models/deepseek_r1

# NVIDIA GPU
python attention_test.py --nvidia

# CPU
python attention_test.py --cpu

# Moore
python attention_test.py --moore

# MetaX
python attention_test.py --metax

# Iluvatar
python attention_test.py --iluvatar
```

### 使用 Slurm（推荐）

```bash
cd /data/users/shihaolin/infini/InfiniLM/test/models/deepseek_r1

# 申请 GPU 资源并运行
srun --gres=gpu:nvidia:1 --cpus-per-task=16 --mem=256G \
  python attention_test.py --nvidia
```

### 带正确性验证

如果要与 transformers 实现对比，需要提供模型路径：

```bash
srun --gres=gpu:nvidia:1 --cpus-per-task=16 --mem=256G \
  python attention_test.py --nvidia --model_path=/data/shared/models/DeepSeek-R1-Layer-3
```

## 模型配置

DeepSeek-R1 (Layer 3) 配置：
```python
hidden_size = 7168
num_attention_heads = 128
num_key_value_heads = 128
q_lora_rank = 1536
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
```

自定义实现配置（在 `mla_module.py` 中）：
```python
dim = 2048  # hidden_size
n_heads = 16
n_kv_heads = 16
q_lora_rank = 0  # 不使用 Q LoRA
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
```

## 输出示例

```
==============================================================================
Test DeepSeek-R1 MLA (Multi-Head Latent Attention)
==============================================================================

==============================================================================
Performance Benchmark - Custom Implementation
==============================================================================
Test Case PREFILL_TESTCASES: {'seqlens': [64, 128, 256, 256], 'pastlens': [512, 0, 0, 256]}
	 WARMUPS=10 RUNS=100, MLA Custom, average latency per batch: 2.22 ms

------------------------------------------------------------------------------

Test DECODE_TESTCASES: {'seqlens': [1, 1, 1, ...], 'pastlens': [50, 50, ..., 400, 400]}
	 WARMUPS=10 RUNS=100, MLA Custom, average throughput: 2224.27 tok/s

==============================================================================
Test Summary
==============================================================================
Custom Implementation - Prefill Latency: 2.22 ms
Custom Implementation - Decode Throughput: 2224.27 tok/s
==============================================================================
```

## 技术要点

### 1. kv_cache + pe_cache 策略 (Absorb 模式)

传统方式（Naive）：
```
Cache: [batch, seq, n_heads, head_dim]
需要存储完整的 K, V
```

Absorb 模式：
```
kv_cache: [batch, seq, kv_lora_rank]  # 压缩的 KV
pe_cache: [batch, seq, qk_rope_head_dim]  # 位置编码
```

优势：
- **显著减少 KV Cache 大小**（从 `n_heads × head_dim` 压缩到 `kv_lora_rank`）
- **适合长序列推理**

### 2. MLA 架构

DeepSeek-V3 的 MLA 使用两阶段投影：
```
Q: hidden_size → q_lora_rank → n_heads × qk_head_dim
KV: hidden_size → kv_lora_rank → n_heads × (qk_nope_head_dim + v_head_dim)
```

这种设计：
- 降低计算复杂度
- 减少参数量
- 保持模型性能

### 3. 注意力计算

```python
# Q 分为两部分
q_nope: 非位置编码部分
q_pe: RoPE 位置编码部分

# Attention Score
score = q_nope @ kv_cache + q_pe @ pe_cache

# 聚合 Value
output = softmax(score) @ kv_cache @ wkv_b
```

## 参考

- 参考实现：`/data/users/shihaolin/infini/model.py` (class MLA)
- 参考测试：`/data/users/shihaolin/infini/InfiniLM/test/models/qwen3_moe/attention_test.py`
- 模型地址：https://huggingface.co/deepseek-ai/DeepSeek-R1
- 本地模型：`/data/shared/models/DeepSeek-R1-Layer-3/`

## 注意事项

1. **模型权重已反量化为 BF16**，无需实现 FP8 计算
2. **必须支持 NVIDIA GPU**，可选支持其他平台
3. **使用 kv_cache + pe_cache 策略**，不使用 naive 模式
4. 性能统计基于**全部请求完成**的总时间
