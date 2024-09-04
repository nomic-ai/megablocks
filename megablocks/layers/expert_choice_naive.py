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

        x_in = torch.einsum('bekn,bnd->bekd', P, x)
        x_e = self.mlp(x_in)
        x_out = torch.einsum('bekn,bek,bekd->bnd', P, expert_weights, x_e)

        return x_out, None
        

class ExpertChoiceMoE(moe.MoE):
    def __init__(self, args: Arguments):
        super().__init__(args)

        self.router = router.ExpertChoiceRouter(args)

    def _init_experts_mlp(self, args: Arguments):
        return ExpertChoiceMLP(args)