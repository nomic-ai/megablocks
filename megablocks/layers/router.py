from megablocks.layers import common
from megablocks.layers.arguments import Arguments
from megablocks.layers import mpu
import torch


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)
_uniform_expert_assignment = _UniformExpertAssignment.apply


class LossFreeRouter(torch.nn.Module):
    """
    Router implementing Loss-Free Balancing for MoE routing.
    Automatically adjusts per-expert biases during training to balance load
    without requiring auxiliary losses.
    """

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device)
        args.init_method(self.layer.weight)

        # Per-expert biases for load balancing
        self.register_buffer(
            'expert_biases',
            torch.zeros(args.moe_num_experts, dtype=common.dtype(args), device=args.device)
        )
        self.bias_update_rate = 0.01  # Can be made configurable via args

    def _sync_biases(self):
        """Synchronize expert biases across DDP processes."""
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.expert_biases)
            self.expert_biases.div_(torch.distributed.get_world_size())

    def _update_biases(self, token_counts, batch_size):
        """Update expert biases based on load violations."""
        if not self.training:
            return

        # Calculate target tokens per expert (average load)
        tokens_per_expert = batch_size * self.args.moe_top_k / self.args.moe_num_experts
        
        # Calculate load violations
        load_violations = token_counts.float() - tokens_per_expert
        
        # Update biases using sign of load violations
        self.expert_biases.add_(
            self.bias_update_rate * load_violations.sign()
        )
        
        # Sync updated biases across processes
        self._sync_biases()

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x, attention_mask=None):
        batch_shape = x.shape[:-1]
        flat_x = x.view(-1, x.shape[-1])
        batch_size = flat_x.shape[0]

        # Get raw routing scores
        raw_scores = self.layer(flat_x)
        
        # Add expert biases to scores
        scores = raw_scores + self.expert_biases
        scores = scores.softmax(dim=-1)

        # Get top-k routing assignments
        expert_weights, expert_indices = self._top_k(scores)

        # Count tokens per expert
        token_counts = torch.bincount(
            expert_indices.view(-1),
            minlength=self.args.moe_num_experts
        )

        # Update biases based on load violations
        self._update_biases(token_counts, batch_size)

        # Reshape outputs to match input batch shape
        scores = scores.view(*batch_shape, -1)
        expert_weights = expert_weights.view(*batch_shape, -1)
        expert_indices = expert_indices.view(*batch_shape, -1)

        return scores, expert_weights, expert_indices


class LearnedRouter(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device)
        args.init_method(self.layer.weight)

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1,keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x, attention_mask=None):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)
        if self.args.moe_expert_choice:
            # Get probability for each token
            bs, sq, _ = x.shape
            capacity = self.args.moe_top_k # Use top k as the capacity to match regular MoEs
            ###scores = self.layer(x).softmax(dim=1) # [batch_size, seq_len, dim] -> [batch_size, seq_len, num_experts]
            # Let experts choose the highest prob tokens
            # k = n * c / e (https://arxiv.org/pdf/2202.09368)
            # where n = tokens (in the paper it seems they even do tokens across batches not just within each batch
            #; could ablate this by folding together bs & sq)
            # c = capacity (1 or 2 in the paper) & e = num_experts
            # [batch_size, seq_len, num_experts] -> [batch_size, k, num_experts] (k is not top k here!!!)
            # expert_weights corresponds to matrix G (e x k) in the paper
            # expert_indices corresponds to matrix I (e x k) in the paper
            ###expert_weights, expert_indices = torch.topk(scores, (capacity * sq) // self.args.moe_num_experts, dim=1)
            # Softmax is still taken along expert dim, not token dim.
            # https://github.com/google/flaxformer/blob/399ea3a85e9807ada653fd0de1a9de627eb0acde/flaxformer/architectures/moe/routing.py#L322C7-L322C66
            # One difference here might be that some do it grouped instead which would be folding bs & sq dims
            # e.g. https://github.com/llm-random/llm-random/blob/c9e50b75cd99f3ae04c3c3bfad0b7f1bf17f6b88/research/conditional/moe_layers/moe_gating.py#L76
            scores = self.layer(x).softmax(dim=-1) # [batch_size, seq_len, num_experts]
            mask = attention_mask.unsqueeze(-1)
            # zero out weights for padding tokens
            scores = scores * mask
            # [batch_size, num_experts, k]
            expert_weights, expert_indices = torch.topk(scores.transpose(1,2), (capacity * sq) // self.args.moe_num_experts, dim=-1)
        else:
            scores = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1)
            expert_weights, expert_indices = self._top_k(scores)

        return scores, expert_weights, expert_indices