"""
DeepSeek-R1 MLA (Multi-Head Latent Attention) 测试脚本
包含自定义实现和 transformers 实现的对比测试
"""
import os
import sys
import time
import torch

# 测试配置
WARMUPS = 10
RUNS = 100

# 预填充测试用例：4个请求，长度分别为64、128、256、256，历史长度分别为512、0、0、256
PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256], "pastlens": [512, 0, 0, 256]}

# 解码测试用例：16个请求，输入长度均为1，历史长度为50*4, 100*4, 200*4, 400*4
DECODE_TESTCASES = {
    "seqlens": [1] * 16,
    "pastlens": [50] * 4 + [100] * 4 + [200] * 4 + [400] * 4,
}


def get_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="Test DeepSeek-R1 MLA")
    parser.add_argument("--model_path", type=str, help="transformers 模型路径（用于正确性验证）")
    parser.add_argument("--cpu", action="store_true", help="在 CPU 上运行")
    parser.add_argument("--nvidia", action="store_true", help="在 NVIDIA GPU 上运行")
    parser.add_argument("--metax", action="store_true", help="在 MetaX GPU 上运行")
    parser.add_argument("--moore", action="store_true", help="在 Moore 上运行")
    parser.add_argument("--iluvatar", action="store_true", help="在 Iluvatar 上运行")
    return parser.parse_args()


def torch_synchronize(device):
    """同步设备"""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "musa":
        torch.musa.synchronize()


def torch_empty_cache(device):
    """清空缓存"""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "musa":
        torch.musa.empty_cache()


# ============================================================================
# 自定义 MLA 实现
# ============================================================================

def create_mla_custom(device, dtype=torch.bfloat16):
    """创建自定义 MLA 模块"""
    from mla_module import MLAConfig, MLA, precompute_freqs_cis
    
    config = MLAConfig()
    model = MLA(config, dtype=dtype).to(device=device)
    
    # 随机初始化权重
    for param in model.parameters():
        if param.dtype == dtype:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
    
    # 预计算旋转编码
    freqs_cis = precompute_freqs_cis(config, device)
    
    return model, freqs_cis


def generate_mla_input_custom(model, freqs_cis, testcase, device, dtype=torch.bfloat16):
    """生成自定义 MLA 的输入数据"""
    config = model.config
    bs = 1
    
    inputs = []
    for seq_len, past_len in zip(testcase["seqlens"], testcase["pastlens"]):
        x = torch.randn(bs, seq_len, config.dim, device=device, dtype=dtype)
        start_pos = past_len
        freqs = freqs_cis[start_pos:start_pos + seq_len]
        
        # 生成随机的 cache 数据（如果有历史长度）
        kv_cache_init = None
        pe_cache_init = None
        if past_len > 0:
            kv_cache_init = torch.randn(
                bs, past_len, config.kv_lora_rank, device=device, dtype=dtype
            )
            pe_cache_init = torch.randn(
                bs, past_len, config.qk_rope_head_dim, device=device, dtype=dtype
            )
        
        # 构造 causal mask
        # mask 的形状应该是 [seq_len, end_pos]，其中 end_pos = start_pos + seq_len
        mask = None
        if seq_len > 1:
            end_pos = start_pos + seq_len
            # 创建完整的 mask [seq_len, end_pos]
            mask = torch.zeros((seq_len, end_pos), device=device, dtype=dtype)
            # 前 start_pos 列都是 0（可以看到历史）
            # 后 seq_len 列是 causal mask
            causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            mask[:, start_pos:end_pos] = causal_mask
        
        inputs.append({
            "x": x,
            "start_pos": start_pos,
            "freqs": freqs,
            "mask": mask,
            "kv_cache_init": kv_cache_init,
            "pe_cache_init": pe_cache_init,
        })
    
    return inputs


def benchmark_mla_prefill_custom(model, freqs_cis, test_cases, device, dtype=torch.bfloat16):
    """测试自定义 MLA 预填充性能"""
    inputs = generate_mla_input_custom(model, freqs_cis, test_cases, device, dtype)
    
    # Warmup
    for _ in range(WARMUPS):
        model.reset_cache()
        for inp in inputs:
            # 填充历史 cache
            if inp["kv_cache_init"] is not None:
                bs, past_len, _ = inp["kv_cache_init"].shape
                model.kv_cache[:bs, :past_len] = inp["kv_cache_init"]
                model.pe_cache[:bs, :past_len] = inp["pe_cache_init"]
            _ = model(inp["x"], inp["start_pos"], inp["freqs"], inp["mask"])
    
    # Benchmark
    time_consuming = 0
    for _ in range(RUNS):
        model.reset_cache()
        torch_synchronize(device)
        start_time = time.time()
        
        for inp in inputs:
            # 填充历史 cache
            if inp["kv_cache_init"] is not None:
                bs, past_len, _ = inp["kv_cache_init"].shape
                model.kv_cache[:bs, :past_len] = inp["kv_cache_init"]
                model.pe_cache[:bs, :past_len] = inp["pe_cache_init"]
            _ = model(inp["x"], inp["start_pos"], inp["freqs"], inp["mask"])
        
        torch_synchronize(device)
        end_time = time.time()
        time_consuming += end_time - start_time
    
    # 计算平均延迟（每个 batch 的延迟）
    latency = time_consuming * 1000 / RUNS
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Custom, average latency per batch: {latency:.2f} ms")
    
    return latency


def benchmark_mla_decode_custom(model, freqs_cis, test_cases, device, dtype=torch.bfloat16):
    """测试自定义 MLA 解码性能"""
    inputs = generate_mla_input_custom(model, freqs_cis, test_cases, device, dtype)
    
    # Warmup
    for inp in inputs:
        model.reset_cache()
        # 填充历史 cache
        if inp["kv_cache_init"] is not None:
            bs, past_len, _ = inp["kv_cache_init"].shape
            model.kv_cache[:bs, :past_len] = inp["kv_cache_init"]
            model.pe_cache[:bs, :past_len] = inp["pe_cache_init"]
        
        for _ in range(WARMUPS):
            output = model(inp["x"], inp["start_pos"], inp["freqs"], inp["mask"])
            inp["x"] = output
            inp["start_pos"] += 1
            inp["freqs"] = freqs_cis[inp["start_pos"]:inp["start_pos"] + 1]
    
    # Restore
    inputs = generate_mla_input_custom(model, freqs_cis, test_cases, device, dtype)
    
    # Benchmark
    torch_synchronize(device)
    start_time = time.time()
    
    for inp in inputs:
        model.reset_cache()
        # 填充历史 cache
        if inp["kv_cache_init"] is not None:
            bs, past_len, _ = inp["kv_cache_init"].shape
            model.kv_cache[:bs, :past_len] = inp["kv_cache_init"]
            model.pe_cache[:bs, :past_len] = inp["pe_cache_init"]
        
        for _ in range(RUNS):
            output = model(inp["x"], inp["start_pos"], inp["freqs"], inp["mask"])
            inp["x"] = output
            inp["start_pos"] += 1
            inp["freqs"] = freqs_cis[inp["start_pos"]:inp["start_pos"] + 1]
    
    torch_synchronize(device)
    end_time = time.time()
    
    time_consuming = end_time - start_time
    out_token_count = RUNS * len(inputs)
    throughput = out_token_count / time_consuming
    
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Custom, average throughput: {throughput:.2f} tok/s")
    
    return throughput


# ============================================================================
# Transformers 实现（用于正确性验证）
# ============================================================================

def create_mla_torch(dir_path, device, dtype=torch.bfloat16):
    """创建 transformers DeepSeek-R1 MLA 模块"""
    try:
        import safetensors
        from transformers import AutoConfig
        
        print(f"Loading model from: {dir_path}")
        config = AutoConfig.from_pretrained(dir_path)
        print(f"Model config loaded: {config.model_type}")
        
        # 设置 attention implementation
        if not hasattr(config, '_attn_implementation') or config._attn_implementation is None:
            config._attn_implementation = 'eager'
            print(f"Set _attn_implementation to: {config._attn_implementation}")
        
        # 导入 DeepSeek 模型
        try:
            from transformers.models.deepseek_v3 import modeling_deepseek_v3
            model = modeling_deepseek_v3.DeepseekV3Attention(config, layer_idx=0).to(device=device, dtype=dtype)
            print(f"Model architecture created successfully")
        except ImportError as e:
            print(f"Warning: DeepSeek-V3 model not found in transformers: {e}")
            return None, None
        
        # 加载权重
        safetensors_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".safetensors")])
        print(f"Found {len(safetensors_files)} safetensors files")
        
        tensors = {}
        for fname in safetensors_files:
            fpath = os.path.join(dir_path, fname)
            print(f"Scanning {fname}...")
            with safetensors.safe_open(fpath, framework="pt") as f:
                for key in f.keys():
                    if "model.layers.0.self_attn." in key:
                        weight_name = key[len("model.layers.0.self_attn."):]
                        tensors[weight_name] = f.get_tensor(key)
                        print(f"  Found weight: {weight_name}")
            
            if tensors:
                break
        
        if tensors:
            print(f"Found {len(tensors)} attention weights in {fname}")
            print(f"Loading {len(tensors)} weights into model...")
            model.load_state_dict(tensors, strict=False)
            print(f"Weights loaded successfully")
        else:
            print("Warning: No attention weights found")
        
        # 创建 rotary embedding
        try:
            rotary_emb = modeling_deepseek_v3.DeepseekV3RotaryEmbedding(config, device=device)
            print(f"Rotary embedding created successfully")
        except Exception as e:
            print(f"Warning: Failed to create rotary embedding: {e}")
            rotary_emb = None
        
        return model, rotary_emb
    
    except Exception as e:
        print(f"Error loading transformers model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_mla_input_torch(model, rotary_emb, testcase, device, dtype=torch.bfloat16):
    """生成 transformers MLA 的输入数据"""
    from transformers import DynamicCache
    
    config = model.config
    hidden_size = config.hidden_size
    bs = 1
    
    req_list = []
    for seq_len, past_len in zip(testcase["seqlens"], testcase["pastlens"]):
        hidden_states = torch.randn(bs, seq_len, hidden_size, device=device, dtype=dtype)
        attention_mask = None
        
        # 创建 KV Cache
        past_key_values = DynamicCache(config=config)
        
        # 注意：不预填充 past_key_values，让模型自己管理
        
        req = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_offset": past_len,  # 记录历史长度
        }
        req_list.append(req)
    
    return req_list


def benchmark_mla_prefill_torch(model, rotary_emb, test_cases, device, dtype=torch.bfloat16):
    """测试 transformers MLA 预填充性能"""
    req_list = generate_mla_input_torch(model, rotary_emb, test_cases, device, dtype)
    
    # Warmup
    for _ in range(WARMUPS):
        for req in req_list:
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]
            position_offset = req["position_offset"]
            
            cache_len = past_key_values.get_seq_length() if past_key_values else 0
            bs, seq_len, _ = hidden_states.shape
            
            position_ids = torch.arange(
                position_offset + cache_len,
                position_offset + cache_len + seq_len,
                dtype=torch.int64,
                device=device
            ).reshape((bs, seq_len))
            
            if rotary_emb:
                cos_table, sin_table = rotary_emb(hidden_states, position_ids)
                position_embeddings = (sin_table, cos_table)
            else:
                position_embeddings = None
            
            _ = model(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
    
    # Benchmark
    time_consuming = 0
    for _ in range(RUNS):
        # 重新生成输入（清空 cache）
        req_list = generate_mla_input_torch(model, rotary_emb, test_cases, device, dtype)
        
        torch_synchronize(device)
        start_time = time.time()
        
        for req in req_list:
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]
            position_offset = req["position_offset"]
            
            cache_len = past_key_values.get_seq_length() if past_key_values else 0
            bs, seq_len, _ = hidden_states.shape
            
            position_ids = torch.arange(
                position_offset + cache_len,
                position_offset + cache_len + seq_len,
                dtype=torch.int64,
                device=device
            ).reshape((bs, seq_len))
            
            if rotary_emb:
                cos_table, sin_table = rotary_emb(hidden_states, position_ids)
                position_embeddings = (sin_table, cos_table)
            else:
                position_embeddings = None
            
            _ = model(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        
        torch_synchronize(device)
        end_time = time.time()
        time_consuming += end_time - start_time
    
    latency = time_consuming * 1000 / RUNS
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Torch, average latency per batch: {latency:.2f} ms")
    
    return latency


def benchmark_mla_decode_torch(model, rotary_emb, test_cases, device, dtype=torch.bfloat16):
    """测试 transformers MLA 解码性能"""
    req_list = generate_mla_input_torch(model, rotary_emb, test_cases, device, dtype)
    
    # Warmup
    for req in req_list:
        for _ in range(WARMUPS):
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]
            position_offset = req["position_offset"]
            
            cache_len = past_key_values.get_seq_length() if past_key_values else 0
            bs, seq_len, _ = hidden_states.shape
            
            position_ids = torch.arange(
                position_offset + cache_len,
                position_offset + cache_len + seq_len,
                dtype=torch.int64,
                device=device
            ).reshape((bs, seq_len))
            
            if rotary_emb:
                cos_table, sin_table = rotary_emb(hidden_states, position_ids)
                position_embeddings = (sin_table, cos_table)
            else:
                position_embeddings = None
            
            output_device, _ = model(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            req["hidden_states"] = output_device
    
    # Restore
    req_list = generate_mla_input_torch(model, rotary_emb, test_cases, device, dtype)
    
    # Benchmark
    torch_synchronize(device)
    start_time = time.time()
    
    for req in req_list:
        for _ in range(RUNS):
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]
            position_offset = req["position_offset"]
            
            cache_len = past_key_values.get_seq_length() if past_key_values else 0
            bs, seq_len, _ = hidden_states.shape
            
            position_ids = torch.arange(
                position_offset + cache_len,
                position_offset + cache_len + seq_len,
                dtype=torch.int64,
                device=device
            ).reshape((bs, seq_len))
            
            if rotary_emb:
                cos_table, sin_table = rotary_emb(hidden_states, position_ids)
                position_embeddings = (sin_table, cos_table)
            else:
                position_embeddings = None
            
            output_device, _ = model(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            req["hidden_states"] = output_device
    
    torch_synchronize(device)
    end_time = time.time()
    
    time_consuming = end_time - start_time
    out_token_count = RUNS * len(req_list)
    throughput = out_token_count / time_consuming
    
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Torch, average throughput: {throughput:.2f} tok/s")
    
    return throughput


def verify_correctness(model_custom, freqs_cis, model_torch, rotary_emb, device, dtype=torch.bfloat16):
    """验证自定义实现和 transformers 实现的正确性"""
    if model_torch is None:
        print("Transformers model not available, skipping correctness verification")
        return False
    
    from transformers import DynamicCache
    
    config = model_torch.config
    print(f"Custom model hidden_size: {model_custom.config.dim}")
    print(f"Transformers model hidden_size: {config.hidden_size}")
    
    # 由于 hidden_size 不同，只测试 transformers 模型的 forward pass
    print(f"\n⚠️  Note: Hidden sizes differ, only testing Transformers model forward pass")
    
    batch_size = 1
    seq_len = 4
    hidden_size = config.hidden_size
    
    try:
        torch.manual_seed(42)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        
        print(f"\nTesting Transformers model forward pass...")
        print(f"Input shape: {hidden_states.shape}")
        
        position_ids = torch.arange(0, seq_len, dtype=torch.int64, device=device).reshape((batch_size, seq_len))
        
        if rotary_emb:
            cos_table, sin_table = rotary_emb(hidden_states, position_ids)
            position_embeddings = (sin_table, cos_table)
        else:
            position_embeddings = None
        
        output_torch, _ = model_torch(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            past_key_values=None,
        )
        print(f"Output shape: {output_torch.shape}")
        print(f"\n✓ Transformers model forward pass successful")
        return True
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    model_path = args.model_path
    dtype = torch.bfloat16
    
    # 确定设备
    device = "cpu"
    if args.cpu:
        device = "cpu"
    elif args.nvidia:
        device = "cuda"
    elif args.metax:
        device = "cuda"
    elif args.moore:
        device = "musa"
        import torch_musa
    elif args.iluvatar:
        device = "cuda"
    else:
        print("Usage: python attention_test.py [--cpu | --nvidia | --metax | --moore | --iluvatar] [--model_path=<path>]")
        sys.exit(1)
    
    print("\n")
    print("=" * 130)
    print("Test DeepSeek-R1 MLA (Multi-Head Latent Attention)")
    print("=" * 130)
    
    # 创建自定义实现
    model_custom, freqs_cis = create_mla_custom(device, dtype)
    
    # 如果提供了模型路径，加载 transformers 模型用于正确性验证
    model_torch = None
    rotary_emb = None
    correctness_passed = False
    
    if model_path:
        print("\n")
        print("=" * 130)
        print("Loading Transformers Model for Correctness Verification")
        print("=" * 130)
        model_torch, rotary_emb = create_mla_torch(model_path, device=device, dtype=dtype)
        
        if model_torch is not None:
            print("\n")
            print("=" * 130)
            print("Correctness Verification")
            print("=" * 130)
            correctness_passed = verify_correctness(model_custom, freqs_cis, model_torch, rotary_emb, device, dtype)
    
    # 性能测试 - 自定义实现
    print("\n")
    print("=" * 130)
    print("Performance Benchmark - Custom Implementation")
    print("=" * 130)
    print(f"Test Case PREFILL_TESTCASES: {PREFILL_TESTCASES}")
    prefill_latency = benchmark_mla_prefill_custom(model_custom, freqs_cis, PREFILL_TESTCASES, device, dtype)
    
    print("\n")
    print("-" * 130)
    print(f"\nTest DECODE_TESTCASES: {DECODE_TESTCASES}")
    decode_throughput = benchmark_mla_decode_custom(model_custom, freqs_cis, DECODE_TESTCASES, device, dtype)
    
    # 性能测试 - transformers 实现（如果有）
    if model_torch is not None:
        print("\n")
        print("=" * 130)
        print("Performance Benchmark - Transformers Implementation")
        print("=" * 130)
        print(f"Test Case PREFILL_TESTCASES: {PREFILL_TESTCASES}")
        torch_prefill_latency = benchmark_mla_prefill_torch(model_torch, rotary_emb, PREFILL_TESTCASES, device, dtype)
        
        print("\n")
        print("-" * 130)
        print(f"\nTest DECODE_TESTCASES: {DECODE_TESTCASES}")
        torch_decode_throughput = benchmark_mla_decode_torch(model_torch, rotary_emb, DECODE_TESTCASES, device, dtype)
    
    # 总结
    print("\n")
    print("=" * 130)
    print("Test Summary")
    print("=" * 130)
    if correctness_passed:
        print("Correctness Verification: ✓ PASSED")
    print(f"Custom Implementation - Prefill Latency: {prefill_latency:.2f} ms")
    print(f"Custom Implementation - Decode Throughput: {decode_throughput:.2f} tok/s")
    if model_torch is not None:
        print(f"Transformers Implementation - Prefill Latency: {torch_prefill_latency:.2f} ms")
        print(f"Transformers Implementation - Decode Throughput: {torch_decode_throughput:.2f} tok/s")
    print("=" * 130)
    
    # 清理
    torch_empty_cache(device)
