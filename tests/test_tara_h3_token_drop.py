import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tara_h3'))
import minimax_h3_token_drop as drop
import minimax_h3_temporal_augmented as augmented
import minimax_h3_temporal_fast as fast

ARGS = (2.0, 0.2, 0.25, 6.0)  # strength, window, falloff, cap


def moving_square(frames=22, h=128, w=192):
    """Static background with one moving bright square: motion is concentrated in a few tokens."""
    x = torch.zeros(frames, h, w, 3)
    x[..., 0] = torch.linspace(0, 0.3, w)  # smooth static gradient
    for t in range(frames):
        x[t, 40:72, 8 + 6 * t:40 + 6 * t] = 1.0
    return x


def test_frame_groups_follow_h3_vae():
    assert drop.frame_groups(6) == [1, 4, 4, 4, 4, 1]
    assert sum(drop.frame_groups(32)) == 107  # 107 frames (17n+5) -> 32 latent frames


def test_keep_mask_keeps_frame_zero_and_the_moving_tokens():
    frames, latent_t, rows_h, rows_w = moving_square(), 6, 4, 6
    keep = drop.keep_mask(frames, latent_t, rows_h, rows_w, drop=0.7)
    n = latent_t * rows_h * rows_w
    assert keep.shape == (n,) and keep.dtype == torch.bool
    assert keep[:rows_h * rows_w].all()  # latent frame 0 is always kept
    assert int(keep.sum()) == round(0.3 * n)
    grid = keep.view(latent_t, rows_h, rows_w)
    assert not grid[1:, 3].any()  # bottom row: static gradient only, dropped
    assert grid[1:, 1].any()      # the square's row (pixels 40-72 -> token row 1) is kept


def test_keep_mask_needs_enough_frames():
    with pytest.raises(ValueError):
        drop.keep_mask(moving_square(frames=10), 6, 4, 6, drop=0.5)


def test_prune_layout_keeps_kept_rows_and_segments():
    pos = torch.arange(30).float()[:, None].repeat(1, 3)
    layout = SimpleNamespace(position_ids=pos, seq_len=30, segments=[(0, 10, 'text'), (10, 20, 'ref'), (20, 30, 'video')])
    keep = torch.ones(30, dtype=torch.bool)
    keep[12:18] = False
    pruned = drop.prune_layout(layout, keep)
    assert pruned.seq_len == 24
    assert pruned.segments == [(0, 10, 'text'), (10, 14, 'ref'), (14, 24, 'video')]
    assert torch.equal(pruned.position_ids, pos[keep])  # kept tokens keep their original RoPE positions
    assert pruned._tara_full is layout and torch.equal(pruned._tara_keep, torch.nonzero(keep).flatten())
    assert layout.seq_len == 30  # the original layout is untouched


def pruned_layout(frames=9, rows=24, text=40, keep_share=0.5, seed=0):
    times = torch.tensor([0.0] + [0.04 + i / 6 for i in range(frames - 1)]).double()  # H3: short first latent frame
    g = torch.Generator().manual_seed(seed)
    src = times.repeat_interleave(rows)
    keep = torch.rand(src.numel(), generator=g) < keep_share
    keep[:rows] = True
    src = src[keep]
    tgt = times.repeat_interleave(rows)
    t = torch.cat([torch.zeros(text, dtype=torch.float64), src, tgt])
    source = torch.zeros(t.numel(), dtype=torch.bool); source[text:text + src.numel()] = True
    target = torch.zeros_like(source); target[text + src.numel():] = True
    return t, source, target


def dense_oracle(q, k, v, t, source, target):
    pen = augmented.penalty((t[:, None] - t[None, :]).abs(), *ARGS)
    bias = torch.where(target[:, None] & source[None, :], -pen, 0.0).float().to(q.device)
    return torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=bias)


@pytest.mark.parametrize('keep_share', [1.0, 0.5, 0.3])
def test_augmented_matches_dense_bias_with_variable_rows_per_frame(keep_share):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if dev == 'cuda' else torch.float32
    t, source, target = pruned_layout(keep_share=keep_share)
    torch.manual_seed(0)
    q, k, v = [torch.randn(1, 2, t.numel(), 128, device=dev, dtype=dtype) for _ in range(3)]
    out = augmented.attention(q, k, v, augmented.LazyPlan(t, source, target, *ARGS))
    exact = dense_oracle(q, k, v, t, source, target)
    plain = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    err = (out.float() - exact).abs()
    assert err.max() < 2e-2
    assert (plain - exact).abs().mean() > 5 * err.mean()  # the bias is really applied


@pytest.mark.skipif(not torch.cuda.is_available(), reason='fused kernel needs CUDA')
@pytest.mark.parametrize('keep_share', [0.5, 0.3])
def test_fused_per_row_times_match_exact_bias_after_drop(keep_share):
    fused = pytest.importorskip('minimax_h3_temporal_triton')
    t, source, target = pruned_layout(frames=9, rows=64, text=96, keep_share=keep_share)
    plan = augmented.LazyPlan(t, source, target, *ARGS)
    torch.manual_seed(0)
    q, k, v = [torch.randn(1, 4, t.numel(), 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    times, sa, sb, ta, tb = plan.rows(q.device)
    out = fused.attention_rows(q, k, v, times, sa, sb, ta, tb, *ARGS)
    exact = dense_oracle(q, k, v, t, source, target)
    err = (out.float() - exact).abs()
    assert err.mean() < 2e-3 and err.max() < 5e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='fused kernel needs CUDA')
def test_fused_per_row_kernel_equals_the_fixed_grid_path_without_drop():
    fused = pytest.importorskip('minimax_h3_temporal_triton')
    t, source, target = pruned_layout(frames=9, rows=64, text=96, keep_share=1.0)
    pos = (t * 40).float()[:, None].repeat(1, 3)
    plan = fast.make_plan(pos, source, target, *ARGS)
    torch.manual_seed(0)
    q, k, v = [torch.randn(1, 4, t.numel(), 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    grid = fused.attention(q, k, v, plan, *ARGS)
    rows, sa, sb, ta, tb = augmented.LazyPlan(t, source, target, *ARGS).rows(q.device)
    per_row = fused.attention_rows(q, k, v, rows, sa, sb, ta, tb, *ARGS)
    assert (grid.float() - per_row.float()).abs().max() < 1e-2
