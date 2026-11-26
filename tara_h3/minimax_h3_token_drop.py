"""Codec-style pruning of the motion reference's tokens (experimental).

Idea from CodecOPSD (Wang et al., arXiv 2611.31184): a reference token whose content is predictable from the previous
frame (little motion, little residual) is redundant, so it can be left out of the DiT sequence. This is an independent
GPU implementation of that idea, not their code: plain block matching on luma instead of an H.264 encode.

Per 16x16 block and pixel frame: the best match in the previous frame within +-SEARCH px gives a motion length and a
mean |residual|; a block whose zero-motion residual is below SKIP counts as skipped (no motion, no residual). Frame 0
has no previous frame: its residual is the block's mean distance from a local blur (an intra-prediction stand-in).
Per token (32x32 px over the pixel frames of one latent frame): motion = max, residual = max over frames of the mean
over its four blocks. Priority drop keeps the top share by rank(residual) * (1 + rank(motion)); latent frame 0 is
always kept, since it is the only full view of the static scene that the later frames are predicted from.
"""
import torch
import torch.nn.functional as F

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)  # H3 VAE: per 17 pixel frames, one frame then four groups of four
SEARCH = 8
STEP = 2
SKIP = 1.0  # mean |luma diff| (0-255) under which a block is treated as a codec skip


def frame_groups(latent_t):
    return [FRAME_PER_TOKEN[k % 5] for k in range(latent_t)]


def _luma(frames):
    # frames [T, H, W, 3] float in [0, 1] -> [T, 1, H, W] luma 0-255
    rgb = frames[..., :3].permute(0, 3, 1, 2).float() * 255.0
    return (0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])


def block_signals(y, block=16):
    """y [T, 1, H, W] -> motion px and mean |residual| per block, each [T, H/block, W/block]."""
    t, _, h, w = y.shape
    pool = lambda x: F.avg_pool2d(x, block)
    blur = F.avg_pool2d(F.pad(y[:1], (2, 2, 2, 2), mode='replicate'), 5, stride=1)
    motion = torch.zeros(t, h // block, w // block, device=y.device)
    residual = torch.zeros_like(motion)
    residual[0] = pool((y[:1] - blur).abs())[0, 0]
    if t == 1:
        return motion, residual
    cur, prev = y[1:], y[:-1]
    padded = F.pad(prev, (SEARCH,) * 4, mode='replicate')
    best = pool((cur - prev).abs())[:, 0]
    zero = best.clone()
    best_len = torch.zeros_like(best)
    for dy in range(-SEARCH, SEARCH + 1, STEP):
        for dx in range(-SEARCH, SEARCH + 1, STEP):
            if dx == 0 and dy == 0:
                continue
            shifted = padded[:, :, SEARCH + dy:SEARCH + dy + h, SEARCH + dx:SEARCH + dx + w]
            sad = pool((cur - shifted).abs())[:, 0] + 0.25  # small cost so a tie keeps zero motion
            better = sad < best
            best = torch.where(better, sad, best)
            best_len = torch.where(better, torch.full_like(best_len, (dx * dx + dy * dy) ** 0.5), best_len)
    skipped = zero < SKIP
    motion[1:] = torch.where(skipped, torch.zeros_like(best_len), best_len)
    residual[1:] = torch.where(skipped, torch.zeros_like(best), best)
    return motion, residual


def token_signals(frames, latent_t, rows_h, rows_w):
    """frames [N, H, W, 3] -> motion, residual per token [latent_t, rows_h, rows_w] (token = 32x32 px)."""
    groups = frame_groups(latent_t)
    need = sum(groups)
    if frames.shape[0] < need:
        raise ValueError(f'reference_frames has {frames.shape[0]} frames; the motion video needs {need}. '
                         'Connect the same frames that go into Reference to Video.')
    x = frames[:need].permute(0, 3, 1, 2)
    x = F.interpolate(x.float(), size=(rows_h * 32, rows_w * 32), mode='bilinear', align_corners=False, antialias=True)
    m, r = block_signals(_luma(x.permute(0, 2, 3, 1)))
    # 2x2 blocks per token: max motion, mean residual
    m = F.max_pool2d(m[:, None], 2)[:, 0]
    r = F.avg_pool2d(r[:, None], 2)[:, 0]
    starts = torch.tensor([0] + groups[:-1], device=m.device).cumsum(0)
    tm = torch.stack([m[s:s + g].amax(0) for s, g in zip(starts.tolist(), groups)])
    tr = torch.stack([r[s:s + g].amax(0) for s, g in zip(starts.tolist(), groups)])
    return tm, tr


def _rank(x):
    flat = x.reshape(-1).float()
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.arange(flat.numel(), device=flat.device, dtype=flat.dtype)
    # ties share their mean rank
    uniq, inv = torch.unique(flat, return_inverse=True)
    mean_rank = torch.zeros(uniq.numel(), device=flat.device).index_add_(0, inv, ranks) / torch.bincount(inv).float()
    return (mean_rank[inv] / max(flat.numel() - 1, 1)).reshape(x.shape)


def keep_mask(frames, latent_t, rows_h, rows_w, drop, lam=1.0, keep_first=True):
    """bool [latent_t * rows_h * rows_w] in packed row order (t, h, w): True = keep."""
    with torch.no_grad():
        m, r = token_signals(frames, latent_t, rows_h, rows_w)
        score = _rank(r) * (1.0 + lam * _rank(m))
        n = score.numel()
        k = max(1, round((1.0 - drop) * n))
        keep = torch.zeros(n, dtype=torch.bool, device=score.device)
        if keep_first:
            # Latent frame 0 is the only full view of the static scene (the codec's I-frame): always keep it, and
            # spend the rest of the budget on the later frames.
            first = rows_h * rows_w
            keep[:first] = True
            rest = score.reshape(-1)[first:]
            keep[first + torch.argsort(rest, descending=True, stable=True)[:max(0, k - first)]] = True
        else:
            keep[torch.argsort(score.reshape(-1), descending=True, stable=True)[:k]] = True
    return keep.cpu()


def prune_layout(layout, keep_rows):
    """A copy of the packed layout without the rows where keep_rows is False. The model embeds with the full layout
    (`_tara_full`) and gathers the kept rows (`_tara_keep`)."""
    import copy
    pruned = copy.copy(layout)
    idx = torch.nonzero(keep_rows).flatten()
    pruned.position_ids = layout.position_ids[keep_rows]
    pruned.seq_len = int(idx.numel())
    segments, off = [], 0
    for a, b, kind in layout.segments:
        n = int(keep_rows[a:b].sum())
        if n:
            segments.append((off, off + n, kind))
            off += n
    pruned.segments = segments
    pruned._tara_full = layout
    pruned._tara_keep = idx
    return pruned


def install_gather(net):
    """Let MiniMaxH3Model embed with the full layout and keep only the pruned rows. Inactive for ordinary layouts."""
    if getattr(net, '_tara_gather_installed', False):
        return
    original = net._embed_and_pack

    def embed_and_pack(video_x, audio_x, context, layout, payload, transformer_options):
        full = getattr(layout, '_tara_full', None)
        if full is None:
            return original(video_x, audio_x, context, layout, payload, transformer_options)
        h = original(video_x, audio_x, context, full, payload, transformer_options)
        return h.index_select(0, layout._tara_keep.to(h.device))

    net._embed_and_pack = embed_and_pack
    net._tara_gather_installed = True
