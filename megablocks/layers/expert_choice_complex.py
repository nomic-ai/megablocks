import numpy as np
import torch.nn.functional as F
from megablocks.layers import dmoe
from megablocks.layers import router
from megablocks.layers import glu
import megablocks.ops as ops
from megablocks.layers.arguments import Arguments
import torch



class ExpertChoiceMLP(dmoe.ParallelDroplessMLP):
    def __init__(self, args: Arguments):
        super().__init__(args)

    def indices_and_padded_bins(self, x, top_experts):
        _, sl, _ = x.shape
        bs, num_experts, top_k = top_experts.shape
        device = top_experts.device

        # Convert (bs, sl) indices to global indices
        batch_offset = torch.arange(bs, device=device).unsqueeze(1).unsqueeze(2) * sl
        global_indices = top_experts + batch_offset
        flat_global_indices = global_indices.reshape(-1)
        # Create bin_ids (expert ids for each selected token)
        # expert_ids are [0 * top_k, 1 * top_k, ..., (num_experts - 1) * top_k] for each row
        expert_ids_flat = torch.arange(num_experts, device=device).repeat(bs * top_k)
        # we want expert_ids to be grouped by expert so we sort them
        bin_ids, expert_ids_indices = ops.sort(expert_ids_flat, self.sort_end_bit)
        # group by expert_id
        indices = flat_global_indices[expert_ids_indices]

        # For Expert Choice, tokens_per_expert is fixed
        tokens_per_expert = torch.full((num_experts,), bs * top_k, dtype=torch.int32, device=device)

        # Round the token counts up to the block size used in the matrix multiplications
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)

        # Calculate the bin bounds for the sorted tokens
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)

        return indices, bin_ids, bins, padded_bins, tokens_per_expert, expert_ids_indices

    def sparse_forward_once(self, x, expert_weights, top_experts):
        bs, sl, hs = x.shape
        
        indices, bin_ids, bins, padded_bins, tokens_per_expert, reverse_indices = \
            self.indices_and_padded_bins(x, top_experts)

        # Flatten x and gather the selected tokens
        x_flat = x.reshape(-1, hs)
        x_gathered = ops.padded_gather(
            x_flat,
            indices,
            bin_ids,
            bins,
            padded_bins,
            1
        )
        # Create the sparse matrix topology
        topo = self.topology(x_gathered, padded_bins)

        # Perform the expert computation
        x_e = self.mlp(x_gathered, topo)

        # Flatten and sort expert_weights
        expert_weights_flat = expert_weights.reshape(-1)
        expert_weights_sorted = expert_weights_flat[reverse_indices]
        # Un-route the data for the MoE output
        x_out = ops.padded_scatter(
            x_e,
            indices,
            bin_ids,
            expert_weights_sorted,
            bins,
            padded_bins,
            1
        )

        return x_out.reshape(bs, sl, hs), tokens_per_expert
        

class ExpertChoiceMoE(dmoe.dMoE):
    def __init__(self, args: Arguments):
        super().__init__(args)

        self.router = router.ExpertChoiceRouter(args)

    def _init_experts_mlp(self, args: Arguments):
        return ExpertChoiceMLP(args)