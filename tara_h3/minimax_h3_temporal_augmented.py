"""Exact TARA bias in one ordinary attention pass, for any number of rows per frame.

The bias depends only on the frames of the query and the key, so it factors through extra head dimensions:
a target query gets a one-hot of its frame, a motion key gets its bias column (one value per target frame), every
other row gets zeros. Then q'.k' * scale = q.k * scale + bias(frame_q, frame_k) with the original scale, and one
FlashAttention call computes the biased attention. V is zero padded and the extra output columns are dropped.

Unlike the partition and one-pass kernels this does not need a fixed spatial grid per frame, so it also runs on a
motion video whose tokens were pruned (reference_token_drop). The cost is a larger head dimension
(128 + frames, rounded up to a multiple of 16); the bias is carried in BF16/FP16, a relative error of ~0.4%.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class Augment:
    q_extra: torch.Tensor  # [S, E]
    k_extra: torch.Tensor  # [S, E]
    frames: int
    head_dim: int


class LazyPlan:
    """Holds the layout; builds the extra columns on first use for the call's head dim, dtype and device."""
    def __init__(self, times, source, target, strength, window, falloff, cap):
        self.args = (times, source, target, strength, window, falloff, cap)
        self.built = {}

    def rows(self, device):
        """(per-row seconds float32 on device, source start/stop, target start/stop) for the per-row fused kernel."""
        if getattr(self, '_rows', None) is None or self._rows[0].device != device:
            times, source, target = self.args[:3]
            si, ti = torch.nonzero(source.cpu()).flatten(), torch.nonzero(target.cpu()).flatten()
            sa, sb, ta, tb = int(si[0]), int(si[-1]) + 1, int(ti[0]), int(ti[-1]) + 1
            if si.numel() != sb - sa or ti.numel() != tb - ta:
                raise RuntimeError('per-row fused kernel needs contiguous motion and target rows')
            self._rows = (times.float().to(device), sa, sb, ta, tb)
        return self._rows

    def get(self, q):
        key = (q.shape[-1], q.dtype, q.device)
        if key not in self.built:
            self.built = {key: build(*self.args, q.shape[-1], q.dtype, q.device)}
        return self.built[key]


def penalty(dt, strength, window, falloff, cap):
    return (strength * (torch.relu(dt - window) / falloff).square()).clamp(max=cap)


def build(times, source, target, strength, window, falloff, cap, head_dim, dtype, device, multiple=16):
    """times [S] seconds; source, target bool [S]. Returns the extra q/k columns."""
    times = times.double().cpu()
    source, target = source.cpu(), target.cpu()
    frame_times = torch.unique(times[target])
    n = frame_times.numel()
    width = -(-(head_dim + n) // multiple) * multiple - head_dim
    scale = head_dim ** -0.5
    q_extra = torch.zeros(times.numel(), width, dtype=torch.float64)
    k_extra = torch.zeros_like(q_extra)
    ti = torch.nonzero(target).flatten()
    q_extra[ti, torch.searchsorted(frame_times, times[ti])] = 1.0
    si = torch.nonzero(source).flatten()
    dt = (times[si][:, None] - frame_times[None, :]).abs()
    # SDPA multiplies q'.k' by the original scale; divide it out here so the logit gains exactly -penalty.
    k_extra[si, :n] = -penalty(dt, strength, window, falloff, cap) / scale
    return Augment(q_extra.to(device=device, dtype=dtype), k_extra.to(device=device, dtype=dtype), n, head_dim)


def attention(q, k, v, aug):
    """q, k, v [B, H, S, D] -> [B, H, S, D] with the TARA bias."""
    b, h, s, d = q.shape
    if isinstance(aug, LazyPlan):
        aug = aug.get(q)
    if d != aug.head_dim or aug.q_extra.shape[0] != s or k.shape[2] != s:
        raise ValueError('augmented temporal attention: sequence or head dimension does not match the plan')
    qe = aug.q_extra.to(q.dtype)[None, None].expand(b, h, -1, -1)
    ke = aug.k_extra.to(k.dtype)[None, None].expand(b, h, -1, -1)
    qa = torch.cat((q, qe), -1)
    ka = torch.cat((k, ke), -1)
    va = F.pad(v, (0, qe.shape[-1]))
    out = F.scaled_dot_product_attention(qa, ka, va, scale=d ** -0.5)
    return out[..., :d]
