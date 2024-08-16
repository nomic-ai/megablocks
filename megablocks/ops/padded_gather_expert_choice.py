import torch
from megablocks.backend import kernels
from stk.backend.autocast import custom_fwd, custom_bwd


class PaddedGatherExpertChoiceOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bin_ids, bins, padded_bins, top_k):
        ctx.save_for_backward(indices, bin_ids, bins, padded_bins)
        ctx.top_k = top_k
        return kernels.padded_gather_expert_choice(
            x, indices, bin_ids, None, bins, padded_bins, top_k)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()

        # we have to use this otherwise we get nans in the backward pass
        indices, bin_ids, bins, padded_bins = ctx.saved_tensors
        out = kernels.padded_scatter_expert_choice(
            grad, indices, bin_ids, None, bins, padded_bins, ctx.top_k)
        return out, None, None, None, None, None

padded_gather_expect_choice = PaddedGatherExpertChoiceOp.apply
