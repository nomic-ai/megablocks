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
        x = x.view(-1, x.shape[-1])
        n, _ = x.shape

        P = F.one_hot(top_experts, num_classes=n).to(x.dtype)

        x_in = torch.einsum('ekn,nd->ekd', P, x)

        x_e = self.mlp(x_in)

        x_out = torch.einsum('ekn,ek,ekd->nd', P, expert_weights, x_e)

        return x_out, None
        

class ExpertChoiceMoE(dmoe.dMoE):
    def __init__(self, args: Arguments):
        super().__init__(args)

        self.router = router.ExpertChoiceRouter(args)

    def _init_experts_mlp(self, args: Arguments):
        return ExpertChoiceMLP(args)