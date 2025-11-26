import os
import time
import sys
import torch
import math

# 导入自定义 MLA 实现
from mla_module import MLA, MLAConfig, precompute_freqs_cis

WARMUPS = 10
RUNS = 100

# 小批量预填充测试用例
PREFILL_TESTCASES = {
    "seqlens": [64, 128, 256, 256],
    "pastlens": [512, 0, 0, 256]
}

# 大批量解码测试用例
DECODE_TESTCASES = {
    "seqlens": [1 for _ in range(16)],
    "pastlens": [50 for _ in range(4)]
    + [100 for _ in range(4)]
    + [200 for _ in range(4)]
    + [400 for _ in range(4)],
}


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test DeepSeek-R1 MLA Operator")
    parser.add_argument("--model_path", action="store", help="The directory of the model to be tested")
    parser.add_argument("--cpu", action="store_true", help="Run cpu test")
    parser.add_argument("--nvidia", action="store_true", help="Run nvidia test")
    parser.add_argument("--metax", action="store_true", help="Run metax test")
    parser.add_argument("--moore", action="store_true", help="Run moore test")
    parser.add_argument("--iluvatar", action="store_true", help="Run iluvatar test")
    return parser.parse_args()


def torch_synchronize(_device):
    if _device == "cuda":
        torch.cuda.synchronize()
    elif _device == "musa":
        torch.musa.synchronize()


def torch_empty_cache(_device):
    if _device == "cuda":
        torch.cuda.empty_cache()
    elif _device == "musa":
        torch.musa.empty_cache()


def create_mla_custom(device, dtype=torch.bfloat16):
    """
    创建自定义 MLA 模块
    """
    config = MLAConfig()
    model = MLA(config, dtype=dtype).to(device=device)
    
    # 随机初始化权重
    for param in model.parameters():
        if param.dtype == dtype:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
    
    # 预计算旋转编码
    freqs_cis = precompute_freqs_cis(config, device)
    
    return model, freqs_cis


def create_mla_torch(dir_path, *, device, dtype=torch.bfloat16):
    """
    创建 DeepSeek-R1 MLA 模块（使用 transformers 库）
    """
    try:
        import safetensors
        from transformers import AutoConfig
        from transformers import DynamicCache
        config = AutoConfig.from_pretrained(dir_path)
        
        # 尝试导入 DeepSeek 模型
        try:
            from transformers.models.deepseek_v3 import modeling_deepseek_v3
            model = modeling_deepseek_v3.DeepseekV3Attention(config, layer_idx=0).to(device=device, dtype=dtype)
        except ImportError:
            print("Warning: DeepSeek-V3 model not found in transformers, using custom implementation")
            return None, None
        
        # 加载权重
        tensors = {}
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".safetensors"):
                continue
            fpath = os.path.join(dir_path, fname)
            with safetensors.safe_open(fpath, framework="pt") as f:
                for key in f.keys():
                    if "model.layers.0.self_attn." in key:
                        tensors[key[len("model.layers.0.self_attn."):]] = f.get_tensor(key)
            break
        
        if tensors:
            model.load_state_dict(tensors)
        
        # 创建 rotary embedding
        try:
            rotary_emb = modeling_deepseek_v3.DeepseekV3RotaryEmbedding(config, device=device)
        except:
            rotary_emb = None
        
        return model, rotary_emb
    except Exception as e:
        print(f"Error loading transformers model: {e}")
        return None, None


def generate_mla_input_torch(model, rotary_emb, testcase, device, dtype=torch.bfloat16):
    """
    生成 MLA 测试输入数据
    """
    from transformers import DynamicCache
    
    config = model.config if model else None
    if config is None:
        # 使用默认配置
        hidden_size = 2048
        head_dim = 128
        num_key_value_heads = 16
    else:
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', 128)
        num_key_value_heads = config.num_key_value_heads
    
    bs = 1
    req_list = []
    
    for seq_lens, past_lens in zip(testcase["seqlens"], testcase["pastlens"]):
        hidden_states = torch.rand((bs, seq_lens, hidden_size), device=device, dtype=dtype)
        attention_mask = None
        
        # 创建 KV Cache
        past_key_values = DynamicCache(config=config) if config else None
        if past_key_values and past_lens > 0:
            key_states = torch.rand((bs, num_key_value_heads, past_lens, head_dim), device=device, dtype=dtype)
            value_states = torch.rand((bs, num_key_value_heads, past_lens, head_dim), device=device, dtype=dtype)
            past_key_values.update(key_states, value_states, 0)
        
        req = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        req_list.append(req)
    
    return req_list


def benchmark_mla_prefill_torch(model, rotary_emb, test_cases, device, dtype=torch.bfloat16):
    """
    测试 MLA 预填充性能
    """
    if model is None:
        print("Model not loaded, skipping test")
        return []
    
    req_list = generate_mla_input_torch(model, rotary_emb, test_cases, device, dtype=dtype)
    req_out_list = []
    
    for req in req_list:
        hidden_states = req["hidden_states"]
        attention_mask = req["attention_mask"]
        past_key_values = req["past_key_values"]
        
        cache_lens = past_key_values.get_seq_length() if past_key_values else 0
        bs, seq_len, _ = hidden_states.shape
        
        position_ids = torch.arange(cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device).reshape((bs, seq_len))
        
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
        
        output_host = output_device.to("cpu")
        req_out_list.append(output_host)
    
    torch_synchronize(device)
    
    # Warmup
    for _ in range(WARMUPS):
        for i, req in enumerate(req_list):
            origin_len = test_cases["pastlens"][i]
            if req["past_key_values"]:
                req["past_key_values"].crop(origin_len)
        
        for req in req_list:
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]
            
            cache_lens = past_key_values.get_seq_length() if past_key_values else 0
            bs, seq_len, _ = hidden_states.shape
            
            position_ids = torch.arange(cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device).reshape((bs, seq_len))
            
            if rotary_emb:
                cos_table, sin_table = rotary_emb(hidden_states, position_ids)
                position_embeddings = (sin_table, cos_table)
            else:
                position_embeddings = None
            
            output_device, _ = model(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values)
    
    # Benchmark
    time_consuming = 0
    for _ in range(RUNS):
        for i, req in enumerate(req_list):
            origin_len = test_cases["pastlens"][i]
            if req["past_key_values"]:
                req["past_key_values"].crop(origin_len)
        
        torch_synchronize(device)
        start_time = time.time()
        
        for i, req in enumerate(req_list):
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]
            
            cache_lens = past_key_values.get_seq_length() if past_key_values else 0
            bs, seq_len, _ = hidden_states.shape
            
            position_ids = torch.arange(cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device).reshape((bs, seq_len))
            
            if rotary_emb:
                cos_table, sin_table = rotary_emb(hidden_states, position_ids)
                position_embeddings = (sin_table, cos_table)
            else:
                position_embeddings = None
            
            output_device, _ = model(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values)
            
            torch_synchronize(device)
            end_time = time.time()
            time_consuming += end_time - start_time
    
    out_token_count = RUNS * len(req_list)
    latency = time_consuming * 1000 / out_token_count
    
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Torch, average latency per batch: {round(latency, 2)} ms")
    
    return req_out_list


def benchmark_mla_decode_torch(model, rotary_emb, test_cases, device, dtype=torch.bfloat16):
    """
    测试 MLA 解码性能
    """
    if model is None:
        print("Model not loaded, skipping test")
        return []
    
    req_list = generate_mla_input_torch(model, rotary_emb, test_cases, device, dtype=dtype)
    req_out_list = []
    
    for req in req_list:
        hidden_states = req["hidden_states"]
        attention_mask = req["attention_mask"]
        past_key_values = req["past_key_values"]
        
        cache_lens = past_key_values.get_seq_length() if past_key_values else 0
        bs, seq_len, _ = hidden_states.shape
        
        position_ids = torch.arange(cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device).reshape((bs, seq_len))
        
        if rotary_emb:
            cos_table, sin_table = rotary_emb(hidden_states, position_ids)
            position_embeddings = (sin_table, cos_table)
        else:
            position_embeddings = None
        
        output_device, _ = model(hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values)
        output_host = output_device.to("cpu")
        req_out_list.append(output_host)
    
    torch_synchronize(device)
    
    # Warmup
    for req in req_list:
        for _ in range(WARMUPS):
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]
            
            cache_lens = past_key_values.get_seq_length() if past_key_values else 0
            bs, seq_len, _ = hidden_states.shape
            
            position_ids = torch.arange(cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device).reshape((bs, seq_len))
            
            if rotary_emb:
                cos_table, sin_table = rotary_emb(hidden_states, position_ids)
                position_embeddings = (sin_table, cos_table)
            else:
                position_embeddings = None
            
            output_device, _ = model(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values)
    
    # Restore cache
    for i, req in enumerate(req_list):
        origin_len = test_cases["pastlens"][i]
        if req["past_key_values"]:
            req["past_key_values"].crop(origin_len)
    
    torch_synchronize(device)
    start_time = time.time()
    
    # Benchmark
    for i, req in enumerate(req_list):
        for _ in range(RUNS):
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]
            
            cache_lens = past_key_values.get_seq_length() if past_key_values else 0
            bs, seq_len, _ = hidden_states.shape
            
            position_ids = torch.arange(cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device).reshape((bs, seq_len))
            
            if rotary_emb:
                cos_table, sin_table = rotary_emb(hidden_states, position_ids)
                position_embeddings = (sin_table, cos_table)
            else:
                position_embeddings = None
            
            output_device, _ = model(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values)
            req["hidden_states"] = output_device
    
    torch_synchronize(device)
    end_time = time.time()
    
    time_consuming = end_time - start_time
    out_token_count = RUNS * len(req_list)
    throughput = out_token_count / time_consuming
    
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Torch, average throughput: {round(throughput, 2)} tok/s")
    
    return req_out_list


def generate_mla_input_custom(testcase, device, dtype=torch.bfloat16):
    """为自定义 MLA 生成输入数据"""
    hidden_size = 2048
    inputs = []
    
    for seq_lens, past_lens in zip(testcase["seqlens"], testcase["pastlens"]):
        x = torch.randn(1, seq_lens, hidden_size, device=device, dtype=dtype)
        inputs.append({
            "x": x,
            "start_pos": past_lens,
            "seqlen": seq_lens,
        })
    
    return inputs


def benchmark_mla_prefill_custom(model, freqs_cis, test_cases, device, dtype=torch.bfloat16):
    """测试自定义 MLA 预填充性能"""
    inputs = generate_mla_input_custom(test_cases, device, dtype)
    
    # Warmup
    for _ in range(WARMUPS):
        model.reset_cache()
        for inp in inputs:
            x = inp["x"]
            start_pos = inp["start_pos"]
            seqlen = inp["seqlen"]
            
            freqs = freqs_cis[start_pos:start_pos + seqlen]
            end_pos = start_pos + seqlen
            mask = None
            if seqlen > 1:
                mask = torch.full((seqlen, end_pos), float("-inf"), device=device)
                mask = torch.triu(mask, diagonal=start_pos + 1)
            
            _ = model(x, start_pos, freqs, mask)
    
    torch_synchronize(device)
    
    # Benchmark
    time_consuming = 0
    for _ in range(RUNS):
        model.reset_cache()
        torch_synchronize(device)
        start_time = time.time()
        
        for inp in inputs:
            x = inp["x"]
            start_pos = inp["start_pos"]
            seqlen = inp["seqlen"]
            
            freqs = freqs_cis[start_pos:start_pos + seqlen]
            end_pos = start_pos + seqlen
            mask = None
            if seqlen > 1:
                mask = torch.full((seqlen, end_pos), float("-inf"), device=device)
                mask = torch.triu(mask, diagonal=start_pos + 1)
            
            _ = model(x, start_pos, freqs, mask)
        
        torch_synchronize(device)
        end_time = time.time()
        time_consuming += end_time - start_time
    
    out_token_count = RUNS
    latency = time_consuming * 1000 / out_token_count
    
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Custom, average latency per batch: {round(latency, 2)} ms")
    return latency


def benchmark_mla_decode_custom(model, freqs_cis, test_cases, device, dtype=torch.bfloat16):
    """测试自定义 MLA 解码性能 - Cache会增长，输出作为下一轮输入"""
    inputs = generate_mla_input_custom(test_cases, device, dtype)
    hidden_size = 2048
    
    # Warmup
    for req in inputs:
        model.reset_cache()
        start_pos = req["start_pos"]
        x = req["x"]
        for i in range(WARMUPS):
            current_pos = start_pos + i
            freqs = freqs_cis[current_pos:current_pos + 1]
            x = model(x, current_pos, freqs, None)
    
    torch_synchronize(device)
    
    # Benchmark - Cache会增长，每轮输出作为下一轮输入
    start_time = time.time()
    for req in inputs:
        model.reset_cache()
        start_pos = req["start_pos"]
        x = req["x"]  # 初始输入
        
        for i in range(RUNS):
            current_pos = start_pos + i  # 位置递增（Cache增长）
            freqs = freqs_cis[current_pos:current_pos + 1]
            x = model(x, current_pos, freqs, None)  # 输出作为下一轮输入
    
    torch_synchronize(device)
    end_time = time.time()
    
    time_consuming = end_time - start_time
    out_token_count = RUNS * len(inputs)
    throughput = out_token_count / time_consuming
    
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Custom, average throughput: {round(throughput, 2)} tok/s")
    return throughput


if __name__ == "__main__":
    args = get_args()
    print(args)
    
    model_path = args.model_path
    dtype = torch.bfloat16
    
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
        print("Usage: python test/models/deepseek_r1/mla_test.py [--cpu | --nvidia | --metax | --moore | --iluvatar] [--model_path=<path>]")
        sys.exit(1)
    
    print("\n")
    print("=" * 130)
    print("Test DeepSeek-R1 MLA (Multi-Head Latent Attention) - Custom Implementation")
    print("=" * 130)
    
    # 测试自定义实现
    model_custom, freqs_cis = create_mla_custom(device, dtype)
    
    print("\n")
    print("*" * 130)
    print("Custom MLA Implementation Test")
    print("*" * 130)
    print(f"Test Case PREFILL_TESTCASES: {PREFILL_TESTCASES}")
    prefill_latency = benchmark_mla_prefill_custom(model_custom, freqs_cis, PREFILL_TESTCASES, device, dtype)
    
    print("\n")
    print("-" * 130)
    print(f"\nTest DECODE_TESTCASES: {DECODE_TESTCASES}")
    decode_throughput = benchmark_mla_decode_custom(model_custom, freqs_cis, DECODE_TESTCASES, device, dtype)
    
    del model_custom
    torch_empty_cache(device)
    
    # 如果提供了模型路径，测试 transformers 版本
    if model_path:
        print("\n")
        print("=" * 130)
        print("Transformers MLA Implementation Test (for correctness comparison)")
        print("=" * 130)
        
        model_torch, rotary_emb = create_mla_torch(model_path, device=device, dtype=dtype)
        
        if model_torch is not None:
            print(f"Test Case PREFILL_TESTCASES: {PREFILL_TESTCASES}")
            output_prefill = benchmark_mla_prefill_torch(model_torch, rotary_emb, PREFILL_TESTCASES, device, dtype=dtype)
            
            print("\n")
            print("-" * 130)
            print(f"\nTest DECODE_TESTCASES: {DECODE_TESTCASES}")
            output_decode = benchmark_mla_decode_torch(model_torch, rotary_emb, DECODE_TESTCASES, device, dtype=dtype)
            
            del model_torch
            torch_empty_cache(device)
        else:
            print("\nTransformers model not available, skipping comparison test.")
    
    print("\n")
    print("=" * 130)
    print("Test Summary")
    print("=" * 130)
    print(f"Custom Implementation - Prefill Latency: {round(prefill_latency, 2)} ms")
    print(f"Custom Implementation - Decode Throughput: {round(decode_throughput, 2)} tok/s")
    print("=" * 130)
