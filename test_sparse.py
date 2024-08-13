import torch.nn.functional as F
import torch


x = torch.tensor([
            [[1, 2, 3, 4],  # Token 0
            [5, 6, 7, 8]],  # Token 1
            [[9, 10, 11, 12],  # Token 0
             [13, 14, 15, 16]]  # Token 1
        ], dtype=torch.float32)

top_experts = torch.tensor([
    [[0], [1]],  # Expert 0 and Expert 1 selects tokens 0 and 1
    [[0], [0]]   # Expert 0 and Expert 1 select tokens 0 and 0
])

expert_weights = torch.tensor([
    [[0.6], [0.4]],  # Weights for Expert 0's selections
    [[0.7], [0.2]]   # Weights for Expert 1's selections
])

n = x.shape[1] 

P = F.one_hot(top_experts, num_classes=n).to(x.dtype)

x_in = torch.einsum('bekn,bnd->bekd', P, x)

x_e = x_in

x_out = torch.einsum('bekn,bek,bekd->bnd', P, expert_weights, x_e)
import pdb; pdb.set_trace()
torch.allclose(x_out[1][0], x_e[1][0] * (expert_weights[1][0] + expert_weights[1][1]))

