"""
DeepSeek-R1 MLA Performance Test using Ninetoothed
"""
import os
import time
import sys
import torch
import math
import argparse

# Import ninetoothed MLA implementation
from mla_ninetoothed import MLANinetoothed, MLAConfig, precompute_freqs_cis

WARMUPS = 10
RUNS = 100

# Small batch prefill test case
PREFILL_TESTCASES = {
    "seqlens": [64, 128, 256, 256],
    "pastlens": [512, 0, 0, 256]
}

# Large batch decode test case
DECODE_TESTCASES = {
    "seqlens": [1] * 16,
    "pastlens": [50, 50, 50, 50, 100, 100, 100, 100, 
                 200, 200, 200, 200, 400, 400, 400, 400]
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Test on CPU")
    parser.add_argument("--nvidia", action="store_true", help="Test on NVIDIA GPU")
    parser.add_argument("--metax", action="store_true", help="Test on MetaX")
    parser.add_argument("--moore", action="store_true", help="Test on Moore")
    parser.add_argument("--iluvatar", action="store_true", help="Test on Iluvatar")
    return parser.parse_args()


def torch_synchronize(device):
    """Synchronize device"""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'musa':
        torch.musa.synchronize()


def create_mla_ninetoothed(device, dtype=torch.bfloat16):
    """Create Ninetoothed MLA module"""
    config = MLAConfig()
    model = MLANinetoothed(config, dtype=dtype).to(device=device)
    
    # Random initialize weights
    for param in model.parameters():
        if param.dtype == dtype:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
    
    # Precompute rotary encodings
    freqs_cis = precompute_freqs_cis(config, device)
    
    return model, freqs_cis


def generate_mla_input(test_cases, device, dtype=torch.bfloat16):
    """Generate MLA test input data"""
    hidden_size = 2048
    inputs = []
    
    for seq_len, past_len in zip(test_cases["seqlens"], test_cases["pastlens"]):
        x = torch.rand((1, seq_len, hidden_size), device=device, dtype=dtype)
        inputs.append({
            "x": x,
            "seqlen": seq_len,
            "start_pos": past_len
        })
    
    return inputs


def benchmark_mla_prefill(model, freqs_cis, test_cases, device, dtype=torch.bfloat16):
    """Test MLA prefill performance"""
    inputs = generate_mla_input(test_cases, device, dtype)
    
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
    
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Ninetoothed, average latency per batch: {round(latency, 2)} ms")
    return latency


def benchmark_mla_decode(model, freqs_cis, test_cases, device, dtype=torch.bfloat16):
    """Test MLA decode performance - Cache grows, output becomes next input"""
    inputs = generate_mla_input(test_cases, device, dtype)
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
    
    # Benchmark - Cache grows, output becomes next input
    start_time = time.time()
    for req in inputs:
        model.reset_cache()
        start_pos = req["start_pos"]
        x = req["x"]  # Initial input
        
        for i in range(RUNS):
            current_pos = start_pos + i  # Position increments (Cache grows)
            freqs = freqs_cis[current_pos:current_pos + 1]
            x = model(x, current_pos, freqs, None)  # Output becomes next input
    
    torch_synchronize(device)
    end_time = time.time()
    
    time_consuming = end_time - start_time
    out_token_count = RUNS * len(inputs)
    throughput = out_token_count / time_consuming
    
    print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Ninetoothed, average throughput: {round(throughput, 2)} tok/s")
    return throughput


def main():
    args = parse_args()
    print(args)
    
    # Determine device
    if args.nvidia or (not args.cpu and not args.metax and not args.moore and not args.iluvatar):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.cpu:
        device = torch.device("cpu")
    elif args.metax or args.moore or args.iluvatar:
        device = torch.device("musa" if torch.musa.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    dtype = torch.bfloat16
    
    print("\n" + "="*130)
    print("Test DeepSeek-R1 MLA (Multi-Head Latent Attention) - Ninetoothed Implementation")
    print("="*130)
    
    print("\n" + "*"*130)
    print("Ninetoothed MLA Implementation Test")
    print("*"*130)
    
    # Create model
    model_nt, freqs_cis = create_mla_ninetoothed(device, dtype)
    
    # Prefill test
    print(f"Test Case PREFILL_TESTCASES: {PREFILL_TESTCASES}")
    prefill_latency = benchmark_mla_prefill(model_nt, freqs_cis, PREFILL_TESTCASES, device, dtype)
    
    print("\n" + "-"*130)
    
    # Decode test
    print(f"\nTest DECODE_TESTCASES: {DECODE_TESTCASES}")
    decode_throughput = benchmark_mla_decode(model_nt, freqs_cis, DECODE_TESTCASES, device, dtype)
    
    # Summary
    print("\n" + "="*130)
    print("Test Summary")
    print("="*130)
    print(f"Ninetoothed Implementation - Prefill Latency: {prefill_latency:.2f} ms")
    print(f"Ninetoothed Implementation - Decode Throughput: {decode_throughput:.2f} tok/s")
    print("="*130)


if __name__ == "__main__":
    main()
