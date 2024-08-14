from megablocks import ops
import torch
import triton
import torch.nn.functional as F


def padded_scatter_matmul(x_e, P, expert_weights):
    x_out = torch.einsum('bekn,bek,bekd->bnd', P, expert_weights, x_e)

    return x_out


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['sl'],  # argument names to use as an x-axis for the plot
        x_vals=[2 ** i for i in range(3, 8)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="padded-scatter-dupe-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'hs': 768, 'num_experts': 8, 'top_k': 1},  # values for function arguments not in `x_names` and `y_name`
    ))

def benchmark(sl, hs, num_experts, top_k, provider):
    x = torch.randn((1, sl, hs), requires_grad=True).cuda().half()
    capacity = max(sl // num_experts, 1)

    # Randomly assign tokens to experts, allowing duplicates.
    # do randperm since each expert can only choose a token once but a token can be chosen by multiple experts
    top_experts = torch.stack([torch.randperm(sl)[:capacity] for _ in range(num_experts)]).cuda().long().unsqueeze(0)
    bs, num_experts, top_k = top_experts.shape
    device = top_experts.device
    expert_weights = torch.rand(top_k, num_experts).cuda().half()

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'triton':
        # bin_ids is i * top_k for each expert
        bin_ids = torch.arange(num_experts, device=device).unsqueeze(-1).repeat(1, capacity).view(-1)
        indices = top_experts.view(-1)

        # For Expert Choice, tokens_per_expert is fixed
        tokens_per_expert = torch.full((num_experts,), bs * top_k, dtype=torch.int32, device=device)

        # Round the token counts up to the block size used in the matrix multiplications
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)

        # Calculate the bin bounds for the sorted tokens
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)

        expert_weights = expert_weights.reshape(-1)

        flat_x = x.reshape(-1, hs)
        gathered_x = ops.padded_gather(flat_x, indices, bin_ids, bins, padded_bins, 1)
        ms = triton.testing.do_bench(lambda: ops.padded_scatter_expert_choice(gathered_x, indices, bin_ids, expert_weights, bins, padded_bins, top_k))
    if provider == 'torch':
        P = F.one_hot(top_experts, num_classes=sl).to(x.dtype)
        x_e = torch.einsum('bekn,bnd->bekd', P, x)
        ms = triton.testing.do_bench(lambda: padded_scatter_matmul(x_e, P, expert_weights.unsqueeze(-1)))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == '__main__':
    benchmark.run(show_plots=True, print_data=True, save_path="./padded_scatter_dupe_benchmark.png")