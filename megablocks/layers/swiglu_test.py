import torch.nn.functional as F
import unittest
from functools import partial

from absl.testing import parameterized
from megablocks.layers.arguments import Arguments
from megablocks.layers.glu import SparseGLU, GroupedGLU
from megablocks.layers import dmlp_registry
from megablocks.layers import testing

import torch
import stk
import numpy as np

def test_modules(
        hidden_size,
        ffn_hidden_size,
        mlp_impl='sparse',
        memory_optimized_mlp=False):
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=1,
        moe_top_k=1,
        init_method=init_method,
        memory_optimized_mlp=memory_optimized_mlp,
        mlp_type='glu',
        mlp_impl=mlp_impl,
        fp16=False,
        bf16=True,
        activation_fn=F.silu)

    swiglu = testing.SwiGLU(args)
    dmoe_swiglu = dmlp_registry.get(args)

    dmoe_swiglu.cuda(torch.cuda.current_device()).to(torch.bfloat16)
    swiglu.cuda(torch.cuda.current_device()).to(torch.bfloat16)

    with torch.no_grad():
        swiglu.w1.copy_(dmoe_swiglu.w1.T)
        swiglu.v1.copy_(dmoe_swiglu.v1.T)
        swiglu.w2.copy_(dmoe_swiglu.w2)

    return args, swiglu, dmoe_swiglu

_DENSE_TESTS = (
    (16, 1024, 512),
    (8, 2048, 512),
)

class GLUTest(parameterized.TestCase):

    @parameterized.parameters(*_DENSE_TESTS)
    def testSwiGLU_ForwardGroupedMLP(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, swiglu, dmoe_swiglu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            mlp_impl='grouped')

        expected_out = swiglu(x)
        tokens_per_expert = torch.tensor([bs * sl]).cuda()
        out = dmoe_swiglu(x.view(bs * sl, hs), tokens_per_expert)
        out = out.view(sl, bs, hs)

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

    @parameterized.parameters(*_DENSE_TESTS)
    def testSwiGLU_ForwardGroupedMLP_MemOpt(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, swiglu, dmoe_swiglu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            mlp_impl='grouped',
            memory_optimized_mlp=True)

        expected_out = swiglu(x)
        tokens_per_expert = torch.tensor([bs * sl]).cuda()
        out = dmoe_swiglu(x.view(bs * sl, hs), tokens_per_expert)
        out = out.view(sl, bs, hs)

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

    @parameterized.parameters(*_DENSE_TESTS)
    def testSwiGLU_ForwardSparseMLP(self, bs, sl, hs):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()

        _, swiglu, dmoe_swiglu = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            mlp_impl='sparse')

        expected_out = swiglu(x)
        with torch.no_grad():
            topo = stk.random.mask(bs * sl, hs * 2, 0, blocking=128).cuda()
        out = dmoe_swiglu(x.view(bs * sl, hs), topo)
        out = out.view(sl, bs, hs)

        self.assertSequenceEqual(out.shape, x.shape)
        self.assertSequenceEqual(expected_out.shape, x.shape)
        self.assertTrue(testing.allclose(out, expected_out))

if __name__ == '__main__':
    unittest.main()
