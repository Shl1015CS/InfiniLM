import argparse
import time
from typing import Dict, List, Optional

import torch

from mla_module import MLA, MLAConfig, precompute_freqs_cis as ref_precompute_freqs
from mla_ntops import (
    DeepSeekR1MLAConfig,
    MLANtops,
    create_mla_ntops,
    precompute_freqs_cis,
)

WARMUPS = 10
RUNS = 100

PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256], "pastlens": [512, 0, 0, 256]}
DECODE_TESTCASES = {
    "seqlens": [1] * 16,
    "pastlens": [50] * 4 + [100] * 4 + [200] * 4 + [400] * 4,
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-R1 MLA ntops 测试")
    parser.add_argument("--model_path", type=str, default=None, help="可选的 transformers 模型路径")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nvidia", action="store_true")
    parser.add_argument("--metax", action="store_true")
    parser.add_argument("--moore", action="store_true")
    parser.add_argument("--iluvatar", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def map_dtype(dtype_flag: str):
    if dtype_flag == "fp32":
        return torch.float32
    if dtype_flag == "fp16":
        return torch.float16
    return torch.bfloat16


def resolve_device(args: argparse.Namespace) -> str:
    if args.cpu:
        return "cpu"
    if args.moore:
        import torch_musa  # noqa: F401

        return "musa"
    return "cuda"


def torch_synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "musa":
        torch.musa.synchronize()


def torch_empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "musa":
        torch.musa.empty_cache()


def build_mask(seq_len: int, start_pos: int, dtype: torch.dtype, device: str) -> Optional[torch.Tensor]:
    if seq_len <= 1:
        return None
    end_pos = start_pos + seq_len
    mask = torch.zeros(seq_len, end_pos, dtype=dtype, device=device)
    causal = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask[:, start_pos:end_pos] = torch.triu(causal, diagonal=1)
    return mask


def generate_inputs(
    config: DeepSeekR1MLAConfig,
    freqs_cis: torch.Tensor,
    test_cases: Dict[str, List[int]],
    device: str,
    dtype: torch.dtype,
) -> List[Dict[str, torch.Tensor]]:
    inputs = []
    hidden_size = config.hidden_size
    for seq_len, past_len in zip(test_cases["seqlens"], test_cases["pastlens"]):
        x = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype)
        start_pos = past_len
        freqs = freqs_cis[start_pos : start_pos + seq_len]
        mask = build_mask(seq_len, start_pos, dtype, device)
        kv_cache_init = None
        pe_cache_init = None
        if past_len > 0:
            kv_cache_init = torch.randn(
                1, past_len, config.kv_lora_rank, device=device, dtype=dtype
            )
            pe_cache_init = torch.randn(
                1, past_len, config.qk_rope_head_dim, device=device, dtype=dtype
            )
        inputs.append(
            {
                "x": x,
                "start_pos": start_pos,
                "freqs": freqs,
                "mask": mask,
                "kv_cache_init": kv_cache_init,
                "pe_cache_init": pe_cache_init,
            }
        )
    return inputs


def load_history(module: MLANtops, case: Dict[str, torch.Tensor]) -> None:
    kv_init = case.get("kv_cache_init")
    if kv_init is None:
        return
    pe_init = case.get("pe_cache_init")
    bsz, hist_len, _ = kv_init.shape
    module.kv_cache[:bsz, :hist_len] = kv_init
    module.pe_cache[:bsz, :hist_len] = pe_init


def benchmark_prefill(
    module: MLANtops,
    freqs_cis: torch.Tensor,
    config: DeepSeekR1MLAConfig,
    device: str,
    dtype: torch.dtype,
) -> float:
    cases = generate_inputs(config, freqs_cis, PREFILL_TESTCASES, device, dtype)
    for _ in range(WARMUPS):
        module.reset_cache()
        for case in cases:
            load_history(module, case)
            _ = module(case["x"], case["start_pos"], case["freqs"], case["mask"])

    torch_synchronize(device)
    elapsed = 0.0
    for _ in range(RUNS):
        module.reset_cache()
        torch_synchronize(device)
        start = time.time()
        for case in cases:
            load_history(module, case)
            _ = module(case["x"], case["start_pos"], case["freqs"], case["mask"])
        torch_synchronize(device)
        elapsed += time.time() - start
    latency_ms = elapsed * 1000 / RUNS
    return latency_ms


def benchmark_decode(
    module: MLANtops,
    freqs_cis: torch.Tensor,
    config: DeepSeekR1MLAConfig,
    device: str,
    dtype: torch.dtype,
) -> float:
    cases = generate_inputs(config, freqs_cis, DECODE_TESTCASES, device, dtype)
    for _ in range(WARMUPS):
        module.reset_cache()
        for case in cases:
            load_history(module, case)
            x = case["x"]
            start_pos = case["start_pos"]
            for _ in range(4):
                out = module(x, start_pos, case["freqs"], None)
                x = out
                start_pos += 1
                case["freqs"] = freqs_cis[start_pos : start_pos + 1]

    torch_synchronize(device)
    start = time.time()
    tokens = 0
    for case in cases:
        module.reset_cache()
        load_history(module, case)
        x = case["x"]
        start_pos = case["start_pos"]
        for _ in range(RUNS):
            out = module(x, start_pos, case["freqs"], None)
            x = out
            start_pos += 1
            case["freqs"] = freqs_cis[start_pos : start_pos + 1]
            tokens += x.size(1)
    torch_synchronize(device)
    end = time.time()
    throughput = tokens / (end - start)
    return throughput


def build_reference_model(device: str, dtype: torch.dtype) -> torch.nn.Module:
    cfg = MLAConfig(
        dim=7168,
        max_seq_len=4096 * 4,
        max_batch_size=16,
        n_heads=128,
        n_kv_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_theta=10000.0,
        original_seq_len=4096,
        rope_factor=40,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
    )
    return MLA(cfg, dtype=dtype).to(device=device)


def copy_weights(src: MLANtops, dst: MLA) -> None:
    if src.q_lora_rank > 0:
        dst.wq_a.weight.data.copy_(src.wq_a.weight.data)
        dst.wq_b.weight.data.copy_(src.wq_b.weight.data)
        dst.q_norm.weight.data.copy_(src.q_norm.weight.data)
    else:
        dst.wq.weight.data.copy_(src.wq.weight.data)
    dst.wkv_a.weight.data.copy_(src.wkv_a.weight.data)
    dst.kv_norm.weight.data.copy_(src.kv_norm.weight.data)
    dst.wkv_b.weight.data.copy_(src.wkv_b.weight.data)
    dst.wo.weight.data.copy_(src.wo.weight.data)


def verify_correctness(
    module: MLANtops,
    freqs_cis: torch.Tensor,
    config: DeepSeekR1MLAConfig,
    device: str,
    dtype: torch.dtype,
) -> bool:
    ref_model = build_reference_model(device, dtype)
    copy_weights(module, ref_model)
    ref_freqs = ref_precompute_freqs(ref_model.config, device)

    module.reset_cache()
    ref_model.reset_cache()

    seq_len = 8
    tokens = torch.randn(2, seq_len, config.hidden_size, device=device, dtype=dtype)
    start_pos = 0
    freqs_ntops = freqs_cis[start_pos : start_pos + seq_len]
    freqs_ref = ref_freqs[start_pos : start_pos + seq_len]

    out_ntops = module(tokens, start_pos, freqs_ntops, None)
    out_ref = ref_model(tokens, start_pos, freqs_ref, None)

    return torch.allclose(out_ntops, out_ref, rtol=1e-2, atol=1e-2)


def main() -> None:
    args = get_args()
    device = resolve_device(args)
    dtype = map_dtype(args.dtype)

    model, freqs_cis, config = create_mla_ntops(device=device, dtype=dtype)
    correctness = verify_correctness(model, freqs_cis, config, device, dtype)

    print("Correctness check:", "PASS" if correctness else "FAIL")

    print("\n=== Prefill Benchmark ===")
    prefill_latency = benchmark_prefill(model, freqs_cis, config, device, dtype)
    print(f"Prefill latency: {prefill_latency:.2f} ms")

    print("\n=== Decode Benchmark ===")
    decode_throughput = benchmark_decode(model, freqs_cis, config, device, dtype)
    print(f"Decode throughput: {decode_throughput:.2f} tok/s")

    torch_empty_cache(device)


if __name__ == "__main__":
    main()
