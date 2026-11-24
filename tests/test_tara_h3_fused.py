import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tara_h3'))
import minimax_h3_temporal_fast as fast

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='fused kernel needs CUDA')


def layout(frames=9, rows=64, text=96, pictures=128, first_frame=0.04):
    n = text + pictures + 2 * frames * rows
    sa, sb = text + pictures, text + pictures + frames * rows
    times = torch.tensor([0.0] + [first_frame + i / 6 for i in range(frames - 1)])  # H3: short first latent frame
    pos = torch.zeros(n, 3)
    pos[sa:sb, 0] = (times * 40).repeat_interleave(rows)
    pos[sb:, 0] = (times * 40).repeat_interleave(rows)
    source = torch.zeros(n, dtype=torch.bool); source[sa:sb] = True
    target = torch.zeros(n, dtype=torch.bool); target[sb:] = True
    return n, fast.make_plan(pos, source, target, 2.0, 0.2, 0.25, 6.0)


@pytest.mark.parametrize('strength', [2.0, 0.0])
def test_fused_matches_exact_partitions_on_nonuniform_clock(strength):
    fused = pytest.importorskip('minimax_h3_temporal_triton')
    n, plan = layout()
    if strength == 0.0:
        plan.cap = 0.0
    torch.manual_seed(0)
    q, k, v = [torch.randn(1, 4, n, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    exact = fast.attention(q, k, v, plan)
    out = fused.attention(q, k, v, plan, strength, 0.2, 0.25, 6.0)
    err = (out.float() - exact.float()).abs()
    assert err.mean() < 2e-3 and err.max() < 5e-2
    if strength:
        dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert (dense.float() - exact.float()).abs().mean() > 5 * err.mean()  # the bias is really applied
