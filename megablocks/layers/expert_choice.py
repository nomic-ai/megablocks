import torch.nn.functional as F
from megablocks.layers import dmoe
from megablocks.layers import moe
from megablocks.layers import router
from megablocks.layers import glu
from megablocks.layers.arguments import Arguments
import torch



class ExpertChoiceMLP(moe.ParallelMLP):
    def __init__(self, args: Arguments):
        super().__init__(args)
        self.mlp = glu.GLU(args)

    def forward_once(self, x, expert_weights, top_experts):
        _, sl, _ = x.shape

        P = F.one_hot(top_experts, num_classes=sl).to(x.dtype)

        # x_in = P @ x.unsqueeze(1)
        # x_e = self.mlp(x_in)
        # x_intermediate = expert_weights.unsqueeze(-1) * x_e
        # x_out = torch.einsum('bekn,bekd->bnd', P, x_intermediate)
        # x_out = torch.einsum('bekn,bek,bekd->bnd', P, expert_weights, x_e)

        # x_inter = expert_weights.unsqueeze(-1) * x_e
        # x_out = torch.matmul(P.transpose(-1, -2), x_inter).sum(dim=1)

        x_in = torch.einsum('bekn,bnd->bekd', P, x)
        x_e = self.mlp(x_in)
        x_out = torch.einsum('bekn,bek,bekd->bnd', P, expert_weights, x_e)


        # x_in = torch.zeros(bs, num_experts, expert_capacity, hs, device=x.device, dtype=x.dtype)
        # # select the rows corresponding to the top experts
        # for i in range(bs):
        #     for j in range(num_experts):
        #         experts = top_experts[i, j]
        #         for k in range(expert_capacity):
        #             x_in[i, j, k] = x[i, experts[k]]

        # x_e = self.mlp(x_in)

        # # scatter back to the original positions with the expert weights
        # x_out = torch.zeros(bs, sl, hs, device=x.device, dtype=x.dtype)
        # for i in range(bs):
        #     for j in range(num_experts):
        #         experts = top_experts[i, j]
        #         for k in range(expert_capacity):
        #             x_out[i, experts[k]] += x_e[i, j, k] * expert_weights[i, j, k]
        

        return x_out, None
        

class ExpertChoiceMoE(dmoe.dMoE):
    def __init__(self, args: Arguments):
        super().__init__(args)

        self.router = router.ExpertChoiceRouter(args)

    def _init_experts_mlp(self, args: Arguments):
        return ExpertChoiceMLP(args)