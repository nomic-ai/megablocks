import torch
from functools import partial
import time

from megablocks.layers import expert_choice
from megablocks.layers import expert_choice_naive
from megablocks.layers.arguments import Arguments

torch.random.manual_seed(0)

def benchmark_layer(layer, inputs, num_runs=100):
    # Warm-up run
    _ = layer(inputs)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = layer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    return (end_time - start_time) / num_runs

def run_benchmark(bs, sl, hs, num_experts):
    inputs = torch.randn(bs, sl, hs).to(dtype=torch.bfloat16, device="cuda")

    ffn_hidden_size = hs * 4
    moe_capacity_factor = 1
    tokens_per_expert = (bs * sl) // num_experts

    common_args = {
        'hidden_size': hs,
        'ffn_hidden_size': ffn_hidden_size,
        'moe_num_experts': num_experts,
        'moe_capacity_factor': moe_capacity_factor,
        'moe_top_k': tokens_per_expert,
        'memory_optimized_mlp': False,
        'mlp_type': 'glu',
        'fp16': False,
        'bf16': True
    }

    # Fast layer
    fast_args = Arguments(**common_args, mlp_impl='sparse')
    fast_layer = expert_choice.ExpertChoiceMoE(fast_args).to(dtype=torch.bfloat16, device="cuda")

    slow_args = Arguments(**common_args)
    slow_layer = expert_choice_naive.ExpertChoiceMoE(slow_args).to(dtype=torch.bfloat16, device="cuda")

    # copy over parameters
    weights = {} 
    for name, param in slow_layer.named_parameters():
        if "experts" in name:
            weights[name] = param.clone().reshape(-1, hs).squeeze()
        else:
            weights[name] = param.clone()

    fast_layer.load_state_dict(weights)
    # Benchmark
    fast_time = benchmark_layer(fast_layer, inputs)
    slow_time = benchmark_layer(slow_layer, inputs)
    
    # return fast, slow, speedup of fast over slow
    return fast_time, slow_time, (slow_time / fast_time)

# Define various input sizes to test
configs = [
    # (batch_size, sequence_length, hidden_size, num_experts)
    (4, 4, 64, 4),
    (4, 8, 64, 8),
    (8, 8, 128, 8),
    (16, 16, 256, 8),
    (32, 32, 512, 8),
    (32, 128, 768, 8),
    (64, 128, 768, 8),
    (128, 128, 768, 8),
    (256, 128, 768, 8),
]

print("BS\tSL\tHS\tExperts\tFast(s)\tSlow(s)\tSpeedup")
print("-" * 70)

for bs, sl, hs, num_experts in configs:
    fast_time, slow_time, speedup = run_benchmark(bs, sl, hs, num_experts)
    print(f"{bs}\t{sl}\t{hs}\t{num_experts}\t{fast_time:.6f}\t{slow_time:.6f}\t{speedup:.2f}x")