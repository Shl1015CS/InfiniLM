# DeepSeek-R1 MLA 九齿语言实现

## 方案 A：使用九齿语言实现 MLA 模块

### 文件结构

```
InfiniLM/test/models/deepseek_r1/
├── mla_ninetoothed.py           # 九齿 MLA 实现（使用九齿算子）
├── attention_test_ninetoothed.py # 九齿版本测试脚本
├── mla_module.py                 # 纯 PyTorch MLA 实现（参考）
└── attention_test.py             # 纯 PyTorch 测试脚本（参考）

InfiniCore/ntops/src/ntops/kernels/
└── mla_attention.py              # 九齿 MLA attention kernel（可选，高级）
```

### 实现特点

#### ✅ 九齿算子使用

`mla_ninetoothed.py` 使用了以下九齿算子：

1. **RMS Norm** - `nt_torch.rms_norm()`
   - 用于 Q/KV 的归一化

2. **Matrix Multiplication** - `nt_torch.mm()`
   - 用于线性投影层

3. **Softmax** - `nt_torch.softmax()`
   - 用于注意力权重计算

4. **RoPE** - 可扩展使用九齿 RoPE 算子
   - 旋转位置编码

#### ✅ 自动跨平台支持

九齿语言自动编译到多个硬件平台：
- ✅ NVIDIA GPU (CUDA)
- ✅ Moore 摩尔线程
- ✅ Tianshu 天数智芯
- ✅ Muxi 沐曦

#### ✅ Absorb 模式实现

使用 `kv_cache + pe_cache` 缓存策略：
```python
# 更新缓存
self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

# 计算注意力
scores = (
    torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
    torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
) * self.softmax_scale
```

### 安装依赖

```bash
# 1. 安装 InfiniCore（包含九齿）
cd /home/shl/work_1125/InfiniCore
pip install -e .

# 2. 安装九齿 Python 包
cd ntops
pip install -e .

# 3. 验证安装
python -c "from ntops import torch as nt_torch; print('✓ Ninetoothed installed')"
```

### 运行测试

#### 1. 纯九齿实现测试

```bash
cd /home/shl/work_1125/InfiniLM/test/models/deepseek_r1

# NVIDIA GPU
python attention_test_ninetoothed.py --nvidia

# CPU（调试用）
python attention_test_ninetoothed.py --cpu

# Moore/Tianshu/Muxi
python attention_test_ninetoothed.py --moore
```

#### 2. 对比测试（PyTorch vs 九齿）

```bash
# PyTorch 版本
python attention_test.py --nvidia

# 九齿版本
python attention_test_ninetoothed.py --nvidia

# 对比性能
```

### 预期输出

```
==================================================================
Test DeepSeek-R1 MLA (Multi-Head Latent Attention) - Ninetoothed
==================================================================

******************************************************************
Ninetoothed MLA Implementation Test
******************************************************************
Test Case PREFILL_TESTCASES: {'seqlens': [64, 128, 256, 256], 'pastlens': [512, 0, 0, 256]}
         WARMUPS=10 RUNS=100, MLA Ninetoothed, average latency per batch: X.XX ms

------------------------------------------------------------------
Test DECODE_TESTCASES: {'seqlens': [1, 1, ...], 'pastlens': [50, 50, ...]}
         WARMUPS=10 RUNS=100, MLA Ninetoothed, average throughput: XXXX.XX tok/s

==================================================================
Test Summary
==================================================================
Ninetoothed Implementation - Prefill Latency: X.XX ms
Ninetoothed Implementation - Decode Throughput: XXXX.XX tok/s
==================================================================
```

### 降级机制

如果九齿未安装，代码自动降级到 PyTorch：

```python
try:
    from ntops import torch as nt_torch
    NINETOOTHED_AVAILABLE = True
except ImportError:
    print("Warning: Ninetoothed not available, falling back to PyTorch")
    NINETOOTHED_AVAILABLE = False
```

### 性能优化建议

#### 阶段 1：快速验证（当前）
- ✅ 使用九齿现有算子组合实现
- ✅ 验证功能正确性
- ✅ 获得基准性能

#### 阶段 2：算子融合（可选）
优化关键路径：
```python
# 融合 norm + projection
# 融合 einsum 操作
# 优化 cache 更新
```

#### 阶段 3：自定义 kernel（可选）
使用 `mla_attention.py` 中的九齿 kernel：
```python
# 完全自定义的 MLA attention kernel
# 针对特定硬件优化
```

### 调试技巧

1. **检查九齿是否可用**
```bash
python -c "from ntops import torch as nt_torch; print(nt_torch.__version__)"
```

2. **对比数值正确性**
```python
# 运行 PyTorch 版本
output_pt = model_pytorch(x, ...)

# 运行九齿版本
output_nt = model_ninetoothed(x, ...)

# 对比
diff = torch.abs(output_pt - output_nt).max()
print(f"Max difference: {diff}")
```

3. **性能 Profiling**
```bash
# 使用 PyTorch profiler
python -m torch.utils.bottleneck attention_test_ninetoothed.py --nvidia
```

### 优势总结

| 特性 | PyTorch 实现 | 九齿实现 |
|------|-------------|---------|
| **开发速度** | 快 | 非常快 |
| **跨平台** | 手动适配 | 自动支持 |
| **性能** | 取决于 PyTorch | 九齿编译器优化 |
| **调试** | 容易 | 容易 |
| **可维护性** | 高 | 高 |
| **硬件优化** | 需要专家 | 编译器自动 |

### 下一步

1. **完成功能验证**
   ```bash
   python attention_test_ninetoothed.py --nvidia
   ```

2. **性能对比**
   ```bash
   python attention_test.py --nvidia > pytorch_results.txt
   python attention_test_ninetoothed.py --nvidia > ninetoothed_results.txt
   diff pytorch_results.txt ninetoothed_results.txt
   ```

3. **多平台测试**（如果可用）
   ```bash
   python attention_test_ninetoothed.py --moore
   python attention_test_ninetoothed.py --iluvatar
   ```

4. **性能优化**（可选）
   - Profiling 找出热点
   - 优化关键算子
   - 考虑算子融合

### 常见问题

**Q: 九齿安装失败？**
```bash
# 检查依赖
pip install ninetoothed

# 从源码安装
cd InfiniCore/ntops
pip install -e .
```

**Q: 性能不如 PyTorch？**
- 首先验证功能正确性
- 检查是否使用了九齿算子（设置 NINETOOTHED_AVAILABLE）
- 考虑编译器优化选项

**Q: 如何验证使用了九齿算子？**
```python
# 在代码中添加日志
print(f"Using Ninetoothed: {NINETOOTHED_AVAILABLE}")
```

### 参考资料

- 官方 MLA 实现：https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
- InfiniCore 文档：`/home/shl/work_1125/InfiniCore/README.md`
- 九齿算子示例：`/home/shl/work_1125/InfiniCore/ntops/src/ntops/kernels/`
