import unittest
from absl.testing import parameterized
from megablocks import ops
import numpy as np
import torch

_PADDED_SCATTER_DUPE_TESTS = (
    (4, 2, 2),
    (4, 2, 4),
    (4, 16, 4),
    (128, 16, 2),
    (128, 16, 4),
    (1024, 16, 4),
    (1024, 16, 8),
    (1024, 16, 64),
    (1024, 16, 16),
    (1024, 768, 4),
    (1024, 768, 8),
)

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def padded_scatter_dupe_reference(x, indices, bin_ids, weights, bins, padded_bins, top_k, hs):
    x = x.detach().cpu().numpy()
    indices = _to_numpy(indices)
    bin_ids = _to_numpy(bin_ids)
    weights = _to_numpy(weights)
    bins = _to_numpy(bins)
    padded_bins = _to_numpy(padded_bins)

    out = np.zeros((indices.shape[0] // top_k, hs))
    out_idx = 0
    for i in range(len(bins)):
        in_idx = 0 if i == 0 else padded_bins[i - 1]
        end = bins[i]
        while out_idx < end:
            store_idx = indices[out_idx]
            scale = weights[out_idx]  # Use out_idx instead of store_idx
            store_idx //= top_k

            out[store_idx, :] += scale * x[in_idx, :]
            out_idx += 1
            in_idx += 1
    return torch.from_numpy(out).cuda().half()

class PaddedScatterDupeTest(parameterized.TestCase):

    @parameterized.parameters(*_PADDED_SCATTER_DUPE_TESTS)
    def testPaddedScatterDupe(self, sl, hs, num_experts):
        # Create the data and indices.
        torch.manual_seed(42)
        x = torch.randn((1, sl, hs), requires_grad=True).cuda().half()

        capacity = max(sl // num_experts, 1)

        # Randomly assign tokens to experts, allowing duplicates.
        # do randperm since each expert can only choose a token once but a token can be chosen by multiple experts
        top_experts = torch.stack([torch.randperm(sl)[:capacity] for _ in range(num_experts)]).cuda().int().unsqueeze(0)
        bs, num_experts, top_k = top_experts.shape
        device = top_experts.device

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

        expert_weights = torch.rand(top_k, num_experts).cuda().half()
        expert_weights = expert_weights.reshape(-1)

        flat_x = x.reshape(-1, hs)
        gathered_x = ops.padded_gather(flat_x, indices, bin_ids, bins, padded_bins, 1)
        expected_out = padded_scatter_dupe_reference(gathered_x, indices, bin_ids, expert_weights, bins, padded_bins, 1, hs)

        out = ops.padded_scatter_dupe(
            gathered_x, indices, bin_ids, expert_weights, bins, padded_bins, 1)

        # TODO: check that the gradients are correct
        out.backward(torch.randn_like(out))  # sanity check backward pass
        # Check approximate equality (scatter reduce uses atomics).
        np.testing.assert_allclose(
            _to_numpy(out), _to_numpy(expected_out), rtol=5e-3)

    def testDuplicateIndices(self):
        x = torch.tensor([
            [[1, 2, 3, 4],  # Token 0
            [5, 6, 7, 8]],  # Token 1
            [[9, 10, 11, 12],  # Token 0
             [13, 14, 15, 16]]  # Token 1
        ], dtype=torch.float32).cuda()
        bs, sl, hs = x.shape

        top_experts = torch.tensor([
            [[0], [1]],  # Expert 0 and Expert 1 selects tokens 0 and 1
            [[0], [0]]   # Expert 0 and Expert 1 select tokens 0 and 0
        ]).cuda()

        expert_weights = torch.tensor([
            [[0.6], [0.4]],  # Weights for Expert 0's selections
            [[0.7], [0.2]]   # Weights for Expert 1's selections
        ]).cuda()

        # Randomly assign tokens to experts.
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
        bin_ids, expert_ids_indices = ops.sort(expert_ids_flat)
        # group by expert_id
        indices = flat_global_indices[expert_ids_indices]

        # For Expert Choice, tokens_per_expert is fixed
        tokens_per_expert = torch.full((num_experts,), bs * top_k, dtype=torch.int32, device=device)

        # Round the token counts up to the block size used in the matrix multiplications
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, 128)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)

        # Calculate the bin bounds for the sorted tokens
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)

        expert_weights = expert_weights.reshape(-1)
        expert_weights = expert_weights[expert_ids_indices]
        flat_x = x.reshape(-1, hs)
        gathered_x = ops.padded_gather(flat_x, indices, bin_ids, bins, padded_bins, 1)

        out = ops.padded_scatter_dupe(
            gathered_x, indices, bin_ids, expert_weights, bins, padded_bins, 1)

        expected_out = padded_scatter_dupe_reference(gathered_x, indices, bin_ids, expert_weights.flatten(), bins, padded_bins, 1, hs)

        # Check that duplicates are handled correctly
        self.assertEqual(out.shape, (4, 4))
        
        # The first and fifth weights should be summed for index 0
        expert_weights = _to_numpy(expert_weights).reshape(-1)
        x = _to_numpy(x)
        out = _to_numpy(out)
        expected_out = _to_numpy(expected_out)
        np.testing.assert_allclose(out[2], x[1][0] * (expert_weights[1] + expert_weights[3]), rtol=1e-3)
        np.testing.assert_allclose(out, expected_out, rtol=1e-3)

if __name__ == '__main__':
    unittest.main(failfast=True)