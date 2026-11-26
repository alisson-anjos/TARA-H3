"""Experimental native H3 temporal inference, matching ai-toolkit native_temporal_v1.

Per-model wrappers only: no global ComfyUI monkey patches or extra weights.
"""
import copy
import inspect
import math
import time
import torch
try:
    from . import minimax_h3_reference_binding as binding
    from . import minimax_h3_token_weights as token_weights
    from . import minimax_h3_native_temporal_core as core
    from . import minimax_h3_temporal_fast as fast
    from . import minimax_h3_temporal_kitchen as kitchen
    from . import minimax_h3_temporal_sol as sol
    from . import minimax_h3_sparse_compat as sparse_compat
    from . import minimax_h3_token_drop as token_drop
    from . import minimax_h3_temporal_augmented as augmented
except ImportError:
    import minimax_h3_reference_binding as binding
    import minimax_h3_token_weights as token_weights
    import minimax_h3_native_temporal_core as core
    import minimax_h3_temporal_fast as fast
    import minimax_h3_temporal_kitchen as kitchen
    import minimax_h3_temporal_sol as sol
    import minimax_h3_sparse_compat as sparse_compat
    import minimax_h3_token_drop as token_drop
    import minimax_h3_temporal_augmented as augmented
try:
    from . import minimax_h3_temporal_triton as fused
except ImportError:
    try:
        import minimax_h3_temporal_triton as fused
    except ImportError:  # no triton or sageattention: fused backend falls back to partitions
        fused = None

MARKER = 'bfs_h3_native_temporal_v1'
SOL_BACKEND = 'sol-attn frame bias (experimental)'
BACKENDS = ['flash partitions (no compile)', 'flex (original)', 'comfy kitchen (experimental)', 'sage partitions', 'sage fused (triton)', 'auto (fastest)', 'flash augmented (one pass)']
AUGMENTED = BACKENDS[6]
AUTO = BACKENDS[5]
STRENGTH_SCALES = ['fixed', 'auto (reference resolution)']
# TARA v1.0 was trained at 672x384: a 21x12 grid of 32 px tokens per reference latent frame.
TRAIN_MOTION_ROWS = 252


def effective_strength(strength, rows_per_frame, mode=STRENGTH_SCALES[0]):
    """auto lowers the bias for reference videos with more tokens per frame than in training (never raises it).

    Measured on MiniMax H3 + TARA v1.0, one 704x1280 clip (880 rows/frame), three seeds: auto (2 -> 1.07)
    raised timing correlation 0.55 -> 0.67 on one seed and matched fixed strength 2 on the others, with the
    same identity retention. Opt-in until it is evaluated on more clips.
    """
    if mode != STRENGTH_SCALES[1] or rows_per_frame <= TRAIN_MOTION_ROWS:
        return strength
    return strength * math.sqrt(TRAIN_MOTION_ROWS / rows_per_frame)
# bias_preset -> (bias_block_start, bias_block_end, bias_sigma_min); 'custom' reads the three inputs.
BIAS_PRESETS = {'full': (0, 99, 0.0), 'blocks 0-11 (fast)': (0, 11, 0.0), 'sigma >= 0.90 (fast)': (0, 99, 0.90), 'custom': None}

def dense_backend_selection(override):
    if override is None:
        return None
    # Recognize only ComfyUI's actual ModelPatcher setter wrapper and official
    # registered dense kernels, never arbitrary attention replacements by name.
    if getattr(override, '__module__', '') != 'comfy.model_patcher' or not getattr(override, '__qualname__', '').endswith('set_model_optimized_attention.<locals>.optimized_attention_override'):
        raise ValueError('Native temporal alignment conflicts with another attention replacement. Remove VSA/sparse or unknown attention patches.')
    try:
        from comfy.ldm.modules.attention import get_attention_function
        selected = inspect.getclosurevars(override).nonlocals['optimized_attention']
    except (ImportError, KeyError, TypeError) as exc:
        raise ValueError('Unable to identify the Model Attention Backend selection.') from exc
    for name in ('comfy_kitchen_int8', 'pytorch'):
        if selected is get_attention_function(name, None) and selected is not None:
            return name
    raise ValueError('Unsupported Model Attention Backend for native temporal alignment.')



def temporal_layout(native, refs, keyframes, video_index=1):
    if keyframes:
        raise ValueError('Native temporal mode cannot be combined with Add Guide, first-frame guides or history. Remove those guides.')
    required = ('position_ids', 'segments', 'seq_len', 'signature')
    if any(not hasattr(native, x) for x in required):
        raise RuntimeError('Unsupported H3 PackedLayout API; update ComfyUI.')
    result = copy.copy(native)
    result.position_ids = native.position_ids.clone()
    source = torch.zeros(native.seq_len, dtype=torch.bool)
    target = torch.zeros_like(source)
    ta, tb, _ = next(s for s in native.segments if s[2] == 'video')
    target[ta:tb] = True
    # Native packing is text, each reference's optional audio then video, targets.
    cursor = next(s[1] for s in native.segments if s[2] == 'text')
    videos = 0
    for ref in refs:
        kind = ref['kind']
        if kind == 'image':
            cursor += (ref['latent_h']//2)*(ref['latent_w']//2)
        elif kind == 'audio':
            cursor += 2*ref['ref_audio_t']
        elif kind in ('video', 'video_audio'):
            videos += 1
            cursor += 2*ref.get('ref_audio_t', 0)
            n = ref['latent_t']*(ref['latent_h']//2)*(ref['latent_w']//2)
            if videos == video_index:
                source[cursor:cursor+n] = True
                result.position_ids[cursor:cursor+n, 0] += native.position_ids[ta, 0] - native.position_ids[cursor, 0]
            cursor += n
        else:
            raise ValueError('Unknown native H3 reference kind: '+str(kind))
    if not source.any():
        raise ValueError(f'Native temporal mode requires <Video {video_index}> as a native video reference. Connect Reference to Video positive conditioning.')
    st = torch.unique_consecutive(result.position_ids[source, 0])
    tt = torch.unique_consecutive(result.position_ids[target, 0])
    if st.shape != tt.shape or not torch.allclose(st, tt, atol=1e-9, rtol=0):
        raise ValueError('Native temporal guide and output must have identical temporal grids. Use the same frame window, 24 fps and the same H3 frame count for both (17n+5, e.g. 107, 124 or 141). No automatic motion stretching is applied.')
    return result, source, target


def attention_override(ctx, previous=None, backend=BACKENDS[1], plan=None, log_first=False, sparse_patch=None, dense_backend=None, reference_biases=None, blocks=(0, 10**6), failed=None):
    first=[log_first]
    failed = set() if failed is None else failed
    def partitions(q, k, v, biases=None):
        if backend in (BACKENDS[4], AUTO) and fused is not None and 'fused' not in failed and not biases:
            try:
                return fused.attention(q,k,v,plan,ctx.strength,ctx.window,ctx.falloff,ctx.cap)
            except RuntimeError as exc:
                failed.add('fused')
                print(f'[TARA-H3] sage fused failed ({exc}); falling back to sage partitions.',flush=True)
        extra = {'reference_biases': biases} if biases else {}
        if backend in (BACKENDS[3], BACKENDS[4], AUTO) and 'sage' not in failed:
            try:
                return fast.attention(q,k,v,plan,flash=fast.sage_with_lse,**extra)
            except (ImportError, RuntimeError) as exc:
                failed.add('sage')
                print(f'[TARA-H3] sage partitions failed ({exc}); falling back to flash partitions.',flush=True)
        return fast.attention(q,k,v,plan,**extra)
    def override(original, q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        # Token refiner and other attention paths retain their original backend.
        n = q.shape[-2] if skip_reshape else q.shape[1]
        block = (kwargs.get('transformer_options') or {}).get('block_index', blocks[0])
        if n != ctx.times.shape[1] or k.shape[-2 if skip_reshape else 1] != n or not blocks[0] <= block <= blocks[1]:
            if previous:
                return previous(original, q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                                skip_output_reshape=skip_output_reshape, **kwargs)
            return original(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                            skip_output_reshape=skip_output_reshape, **kwargs)
        if mask is not None:
            raise ValueError('Native temporal attention does not support another attention mask override.')
        if not (ctx.strength and ctx.cap) and not reference_biases:
            # No temporal bias (clock alignment only): the selected attention backend runs unchanged.
            if previous:
                return previous(original, q, k, v, heads, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
            return original(q, k, v, heads, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        if not skip_reshape:
            q, k, v = [x.view(x.shape[0], x.shape[1], heads, -1).transpose(1, 2) for x in (q,k,v)]
        chosen = backend
        reason = None
        if backend == SOL_BACKEND:
            options = kwargs.get('transformer_options') or {}
            reason = sparse_patch.dense_reason(options, n, options.get('block_index'))
            if reason is None:
                import comfy_kitchen as ck
                if q.shape[-1] != 128 or not ck.sol_attn_is_available(q.device):
                    reason = 'sol-attn requires head dimension 128 and an available CUDA kernel'
            if reason is not None:
                sparse_patch.log_once(('temporal_dense', n, reason), f'temporal dense fallback: {reason}')
                chosen = dense_backend
        timing=first[0]
        if timing:
            if reason is not None:
                print(f'[TARA-H3] sol-attn dense fallback: {reason}',flush=True)
            display_backend=chosen
            if chosen in (SOL_BACKEND,BACKENDS[2],AUTO) and sol.native_kernel.api_for(q) is not None:
                display_backend='sol-attn one-pass (native Kitchen)' if chosen==SOL_BACKEND else 'comfy kitchen one-pass (native)'
            elif chosen==AUTO:
                display_backend='auto: sage fused (no Kitchen one-pass binary for this GPU)'
            elif chosen==AUGMENTED and getattr(plan, 'prefer_fused', False) and fused is not None:
                display_backend='sage fused, per-row times (token drop)'
            print(f'[TARA-H3] first attention starting: backend={display_backend}, Q={tuple(q.shape)}, dtype={q.dtype}',flush=True)
            if q.is_cuda: torch.cuda.synchronize(q.device)
            started=time.perf_counter()
        if reference_biases:
            out = partitions(q,k,v,reference_biases)
        elif chosen == AUGMENTED:
            out = None
            if (fused is not None and getattr(plan, 'prefer_fused', False) and 'fused_rows' not in failed
                    and q.shape[-1] == 128):
                try:
                    times, sa, sb, ta, tb = plan.rows(q.device)
                    out = fused.attention_rows(q,k,v,times,sa,sb,ta,tb,ctx.strength,ctx.window,ctx.falloff,ctx.cap)
                except (ImportError, RuntimeError) as exc:
                    failed.add('fused_rows')
                    print(f'[TARA-H3] sage fused (per-row) failed ({exc}); using flash augmented.',flush=True)
            if out is None:
                out = augmented.attention(q,k,v,plan)
        elif chosen == SOL_BACKEND:
            if not getattr(plan, '_sol_active_logged', False):
                passes = 1 if sol.native_kernel.api_for(q) is not None else (plan.frames+1 if ctx.strength and ctx.cap else 1)
                print(f'[TARA-H3] sol-attn ACTIVE: tau={sparse_patch.tau}, exact conditioning keys, {passes} attention pipeline pass(es).',flush=True)
                plan._sol_active_logged = True
            _, sink_q = sparse_patch.sinks(options, n)
            out = sol.attention(q,k,v,plan,ctx,tau=sparse_patch.tau,
                                token_aug=sparse_patch.extra_tokens,sink_q=sink_q)
        elif chosen == BACKENDS[2]:
            out = kitchen.attention(q,k,v,plan,ctx)
        elif chosen == AUTO:
            out = None
            if 'kitchen' not in failed:
                try:
                    out = sol.native_kernel.run('dense',q,k,v,plan,ctx)
                except (ImportError, OSError, RuntimeError) as exc:
                    print(f'[TARA-H3] Kitchen one-pass failed ({exc}); using sage fused.',flush=True)
                if out is None:
                    failed.add('kitchen')
            if out is None:
                out = partitions(q,k,v)
        elif chosen in (BACKENDS[0], BACKENDS[3], BACKENDS[4]):
            out = partitions(q,k,v)
        else:
            out = core.attention(q,k,v,ctx)
        if timing:
            if q.is_cuda: torch.cuda.synchronize(q.device)
            print(f'[TARA-H3] first attention finished in {time.perf_counter()-started:.3f}s',flush=True)
            first[0]=False
        return out if skip_output_reshape else out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[1]*out.shape[3])
    return override


def structure_free_payload(payload, native, video_index):
    """The payload with motion <Video N> (and its audio rows) taken out; text, pictures and keyframes kept."""
    refs, videos = [], 0
    for ref in payload.get('refs', []):
        if ref['kind'] in ('video', 'video_audio'):
            videos += 1
            if videos == video_index:
                continue
        refs.append(ref)
    keyframes = payload.get('keyframes') or []
    return dict(payload, refs=refs,
        cond_video_latents=[k['latent'] for k in keyframes if k.get('latent') is not None]+[r['latent'] for r in refs if 'latent' in r],
        cond_audio_latents=[k['audio_latent'] for k in keyframes if k.get('audio_latent') is not None]+[r['audio_latent'] for r in refs if r.get('audio_latent') is not None],
        layout=type(native)(*native.signature, keyframes=payload.get('keyframes'), refs=refs))


def structure_guided(real, free, guidance):
    """real + (w-1)(real-free): SolarRenderer's structure-guided answer (arXiv 2610.28630, Sec. 3.3), here at inference."""
    if isinstance(real, (list, tuple)):
        return type(real)(structure_guided(r, f, guidance) for r, f in zip(real, free))
    return real + (guidance - 1.0) * (real - free)


def patch_temporal_model(model, video_index=1, strength=2.0, window=0.2, falloff=0.25, cap=6.0, backend=BACKENDS[0], telemetry=None,
                         structure_guidance=1.0, structure_sigma_min=0.0, bias_block_start=0, bias_block_end=10**6, bias_sigma_min=0.0,
                         strength_scale=STRENGTH_SCALES[0], token_drop_share=0.0, reference_frames=None):
    if not 0.0 <= token_drop_share < 1.0: raise ValueError('reference_token_drop must be in [0, 1)')
    if token_drop_share > 0 and reference_frames is None:
        raise ValueError('reference_token_drop needs reference_frames: connect the same frames that go into Reference to Video as <Video N>.')
    if backend not in BACKENDS: raise ValueError("Unknown native temporal attention backend")
    if not math.isfinite(structure_guidance) or structure_guidance < 1.0: raise ValueError('structure_guidance must be >= 1 (1 = off)')
    if backend==BACKENDS[4] and fused is None:
        print('[TARA-H3] sage fused needs triton and sageattention; using sage partitions.',flush=True)
        backend=BACKENDS[3]
    if backend==BACKENDS[3]:
        try:
            import sageattention  # noqa: F401
        except ImportError:
            print('[TARA-H3] sageattention is not installed; using flash partitions.',flush=True)
            backend=BACKENDS[0]
    if backend==BACKENDS[0] and not hasattr(torch.ops.aten, '_scaled_dot_product_flash_attention'):
        raise RuntimeError("This PyTorch build lacks native FlashAttention with LSE. Update PyTorch or select flex (original).")
    cfg = core.options(dict(native_temporal_alignment=True, native_temporal_strength=strength,
                            native_temporal_window_seconds=window, native_temporal_falloff_seconds=falloff,
                            native_temporal_max_penalty=cap))
    if isinstance(video_index, bool) or not isinstance(video_index, int) or video_index < 1:
        raise ValueError('motion_video_index must be a positive integer')
    if model is None:
        raise ValueError('Native temporal mode needs the MODEL input. Route this node MODEL output to the sampler/guider.')
    if model.model_options.get(MARKER):
        raise ValueError('Native temporal alignment is already installed. Use only one native temporal node per model path.')
    net = model.get_model_object('diffusion_model')
    if net.__class__.__name__ != 'MiniMaxH3Model':
        raise ValueError('Native temporal alignment requires a native ComfyUI MiniMaxH3 model.')
    opts = model.model_options.get('transformer_options', {})
    dense_override, sparse_patch = sparse_compat.selection(opts)
    selected_dense_at_setup=dense_backend_selection(dense_override)
    sparse_compat.without_owned_blocks(opts, sparse_patch)
    if backend in (BACKENDS[2], AUTO) or selected_dense_at_setup=='comfy_kitchen_int8' or sparse_patch is not None:
        sol.native_kernel.prepare_model(model)
    patched = model.clone()
    previous = patched.model_options.get('model_function_wrapper')
    cache = [None, None]
    logged = [False]
    plan_cache = [None, None]
    calls = [0]
    reference_cache = [None, None, None, None]
    token_cache = [None, None, None, None]
    free_cache = [None, None]
    drop_cache = [None, None]
    failed_backends = set()
    if token_drop_share > 0:
        token_drop.install_gather(net)

    def pruned(native, payload, layout, source, target, device):
        refs, videos = payload.get('refs', []), 0
        blk = None
        for ref in refs:
            if ref['kind'] in ('video', 'video_audio'):
                videos += 1
                if videos == video_index:
                    blk = ref
        keep_src = token_drop.keep_mask(reference_frames.to(device), blk['latent_t'], blk['latent_h']//2, blk['latent_w']//2, token_drop_share)
        if keep_src.numel() != int(source.sum()):
            raise RuntimeError('reference token drop: mask size does not match the motion rows')
        keep = torch.ones(layout.seq_len, dtype=torch.bool)
        src_idx = torch.nonzero(source).flatten()
        keep[src_idx[~keep_src]] = False
        print(f'[TARA-H3] reference token drop: kept {int(keep_src.sum())}/{keep_src.numel()} motion rows '
              f'({100*(1-keep_src.float().mean()):.0f}% dropped); packed rows {layout.seq_len} -> {int(keep.sum())}', flush=True)
        return token_drop.prune_layout(layout, keep), source[keep], target[keep]

    def wrapper(model_function, args):
        conds = args.get('c', {})
        payload = conds.get('minimax_payload') or {}
        native = payload.get('layout')
        if native is None:
            raise RuntimeError('Native H3 layout missing. Update ComfyUI and use native AV latent conditioning.')
        if cache[0] is not native:
            cache[:] = [native, temporal_layout(native, payload.get('refs', []), payload.get('keyframes'), video_index)]
        layout, source, target = cache[1]
        dropping = token_drop_share > 0
        if dropping:
            if drop_cache[0] is not native:
                dev = args['input'][0].device if isinstance(args['input'], (list, tuple)) else args['input'].device
                drop_cache[:] = [native, pruned(native, payload, layout, source, target, dev)]
            layout, source, target = drop_cache[1]
        frames = torch.unique_consecutive(layout.position_ids[source, 0]).numel()
        bias_strength = effective_strength(strength, int(source.sum()) // max(frames, 1), strength_scale)
        device = args['input'][0].device if isinstance(args['input'], (list, tuple)) else args['input'].device
        to = dict(conds.get('transformer_options', {}))
        incoming = to.get('optimized_attention_override')
        dense_override, sparse_patch = sparse_compat.selection(to)
        selected_dense = dense_backend_selection(dense_override)
        active_backend = (BACKENDS[2] if selected_dense == 'comfy_kitchen_int8' else
                          BACKENDS[0] if selected_dense == 'pytorch' else backend)
        dense_backend = active_backend
        if sparse_patch is not None:
            active_backend = SOL_BACKEND
        to = sparse_compat.without_owned_blocks(to, sparse_patch)
        times = layout.position_ids[:, 0].float()[None].to(device)/40.0
        ctx = core.TemporalContext(times, source[None].to(device), target[None].to(device),
                                  torch.ones_like(source[None], device=device), bias_strength, window, falloff, cap,
                                  torch.tensor([bias_strength,window,falloff,cap],device=device,dtype=torch.float32))
        if dropping:
            # Pruned motion rows: no fixed grid per frame, so the partition kernels do not apply. The bias (if any)
            # runs as one augmented-head-dim FlashAttention pass; strength 0 keeps the selected backend.
            if bias_strength and cap:
                active_backend = AUGMENTED
            if plan_cache[0] is not layout:
                plan_cache[:] = [layout, None]
            if to.get(binding.OPTION) or to.get(token_weights.OPTION) or sparse_patch is not None:
                raise ValueError('reference_token_drop does not support Reference Binding, Token Weights or sparse attention yet.')
        elif active_backend == AUGMENTED:
            if plan_cache[0] is not layout:
                plan_cache[:] = [layout, None]
        elif plan_cache[0] is not layout:
            plan_cache[:] = [layout, fast.make_plan(layout.position_ids,source,target,bias_strength,window,falloff,cap)]
        if active_backend == AUGMENTED and not isinstance(plan_cache[1], augmented.LazyPlan):
            plan_cache[1] = augmented.LazyPlan(layout.position_ids[:, 0].float()/40.0, source, target, bias_strength,
                                               window, falloff, cap)
            # With token drop and a fast backend selected, prefer the per-row Sage fused kernel (INT8, one pass).
            plan_cache[1].prefer_fused = dropping and backend in (AUTO, BACKENDS[3], BACKENDS[4])
        reference_biases=None
        controls=to.get(binding.OPTION)
        if controls:
            if sparse_patch is not None:
                raise ValueError('Reference Attention Binding does not support sparse attention yet. Remove Model Sparse Attention for this experiment.')
            if reference_cache[0] is not layout or reference_cache[1] is not controls or reference_cache[2] != device:
                reference_cache[:] = [layout,controls,device,binding.build_biases(layout,payload.get('refs',[]),plan_cache[1],controls,device)]
            reference_biases=reference_cache[3]
            if reference_biases and active_backend not in (BACKENDS[3], BACKENDS[4], AUTO):
                active_backend=BACKENDS[0]
                if not logged[0]:
                    print('[TARA-H3 reference binding] ACTIVE: exact Picture logit gains, optional target masks; using Flash Partitions (Kitchen/Sol one-pass bypassed).',flush=True)
        token_control=to.get(token_weights.OPTION)
        if token_control:
            if sparse_patch is not None:
                raise ValueError('Token attention weighting does not support Model Sparse Attention yet')
            if token_cache[0] is not layout or token_cache[1] is not token_control or token_cache[2] != device:
                token_cache[:] = [layout,token_control,device,token_weights.attention_biases(layout,plan_cache[1],token_control,device)]
            reference_biases=list(reference_biases or [])+token_cache[3]
            if active_backend not in (BACKENDS[3], BACKENDS[4], AUTO):active_backend=BACKENDS[0]
            if not logged[0]:print('[TARA-H3 token weights] ACTIVE: prompt text-key logit gains, Flash Partitions; visual rows unchanged.',flush=True)
        if float(args['timestep'].flatten()[0]) >= bias_sigma_min:
            to['optimized_attention_override'] = attention_override(ctx,reference_biases=reference_biases, previous=incoming, backend=active_backend, plan=plan_cache[1], log_first=not logged[0], sparse_patch=sparse_patch, dense_backend=dense_backend,
                                                                    blocks=(bias_block_start, bias_block_end), failed=failed_backends)
        copied = dict(args, c=dict(conds, minimax_payload=dict(payload, layout=layout), transformer_options=to))
        if not logged[0]:
            rows = int(source.sum()) // max(frames, 1)
            scale_note = f' (auto from {strength:g}: {rows} reference rows/frame, {TRAIN_MOTION_ROWS} in training)' if bias_strength != strength else ''
            print(f'[TARA-H3] {core.VERSION}: <Video {video_index}>, {int(source.sum())} motion rows; temporal RoPE aligned; soft window={window}s strength={bias_strength:.2f}{scale_note} cap={cap}; images remain native; Add Guide OFF; backend={"sage fused, per-row times (token drop)" if getattr(plan_cache[1], "prefer_fused", False) and fused is not None else active_backend}; packed_rows={layout.seq_len}', flush=True)
            if sparse_patch is not None:
                print('[TARA-H3] sol-attn uses the native one-pass extension when installed; otherwise frame-wise compatibility repeats work. Watch for Kitchen ONE-PASS sol kernel ACTIVE.', flush=True)
            logged[0] = True
        calls[0]+=1
        if telemetry:
            telemetry(dict(phase='evaluating', evaluation=calls[0], backend=active_backend,
                           motion_rows=int(source.sum()), packed_rows=layout.seq_len,
                           duration_seconds=float((layout.position_ids[target, 0].max()-layout.position_ids[target, 0].min())/40.0)))
        started=time.perf_counter()
        result=previous(model_function, copied) if previous else model_function(copied['input'], copied['timestep'], **copied['c'])
        sigma=float(args['timestep'].flatten()[0])
        if structure_guidance > 1.0 and sigma >= structure_sigma_min:
            # Second pass without the motion video and without the temporal patch (no source rows to bias).
            if free_cache[0] is not native:
                free_cache[:] = [native, structure_free_payload(payload, native, video_index)]
                print(f'[TARA-H3] structure guidance w={structure_guidance:g} at sigma >= {structure_sigma_min:g}: one extra pass without <Video {video_index}> per step', flush=True)
            free_args = dict(args, c=dict(conds, minimax_payload=free_cache[1]))
            free = previous(model_function, free_args) if previous else model_function(free_args['input'], free_args['timestep'], **free_args['c'])
            result = structure_guided(result, free, structure_guidance)
        if calls[0]<=2:
            if device.type=='cuda':torch.cuda.synchronize(device)
            print(f'[TARA-H3] model evaluation {calls[0]} finished in {time.perf_counter()-started:.2f}s (includes weights/offload and all blocks)',flush=True)
        if telemetry:
            telemetry(dict(phase='evaluation_complete', evaluation=calls[0],
                           elapsed_seconds=time.perf_counter()-started))
        return result

    patched.set_model_unet_function_wrapper(wrapper)
    patched.model_options[MARKER] = dict(version=core.VERSION, video_index=video_index, backend=backend, **cfg)
    return patched


class TARAH3TemporalAlignment:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',),
            'motion_video_index': ('INT', {'default':1, 'min':1, 'max':100,
                'tooltip':'Native <Video N> supplying motion. Picture numbering is independent.'}),
            'strength': ('FLOAT', {'default':2.0,'min':0.0,'max':20.0,'step':0.1}),
            'window_seconds': ('FLOAT', {'default':0.2,'min':0.0,'max':10.0,'step':0.01}),
            'falloff_seconds': ('FLOAT', {'default':0.25,'min':0.01,'max':10.0,'step':0.01}),
            'max_penalty': ('FLOAT', {'default':6.0,'min':0.0,'max':30.0,'step':0.1})},
            'hidden': {'unique_id': 'UNIQUE_ID'},
            'optional': {'attention_backend': (BACKENDS, {'default': AUTO, 'tooltip':'auto (default): the Kitchen one-pass CUDA kernel when a verified binary exists for this GPU, otherwise sage fused (one SageAttention Triton pass with the bias inside the kernel), then sage partitions, then flash partitions. With Model Sparse Attention (sol-attn) connected, the sol one-pass kernel is used. An official Model Attention Backend Kitchen selection is detected automatically. Kitchen uses an optional one-pass temporal extension when installed, otherwise frame-wise compatibility. Flex is the original implementation. Sage partitions keep the same exact bias with SageAttention kernels (~2x faster attention than flash partitions). Flash augmented: one FlashAttention pass with the bias carried in extra head dimensions (frames + 128); works with any tokens per frame and is used automatically with reference_token_drop.'}),
                'structure_guidance': ('FLOAT', {'default':1.0,'min':1.0,'max':5.0,'step':0.05,
                    'tooltip':'Experimental structure CFG (SolarRenderer): output + (w-1)*(output - output without <Video N>). 1 = off; ~1.5 follows the motion video more. Costs one extra model pass per step where active.'}),
                'structure_sigma_min': ('FLOAT', {'default':0.0,'min':0.0,'max':1.0,'step':0.01,
                    'tooltip':'Structure guidance only at noise levels >= this (structure is laid down at high noise); lower-noise steps skip the extra pass.'}),
                'strength_scale': (STRENGTH_SCALES, {'default': STRENGTH_SCALES[0],
                    'tooltip':'fixed (default): use strength as given. auto: when the reference video has more tokens per frame than in training (672x384), lower the bias as strength*sqrt(252/rows), e.g. 2.0 -> 1.07 for a 704x1280 reference; never raises it. Try auto for references above ~0.4 MP.'}),
                'bias_preset': (list(BIAS_PRESETS), {'default': 'full',
                    'tooltip':'Where the temporal bias is applied. full = every block and step (training default). blocks 0-11 = early blocks only, other blocks run native attention. sigma >= 0.90 = high-noise steps only. custom = bias_block_start/end and bias_sigma_min below.'}),
                'bias_block_start': ('INT', {'default':0,'min':0,'max':99,
                    'tooltip':'First transformer block that receives the temporal bias. Other blocks run the native attention backend (faster).'}),
                'bias_block_end': ('INT', {'default':99,'min':0,'max':99,
                    'tooltip':'Last transformer block that receives the temporal bias (H3 has 50 blocks, 0-49).'}),
                'bias_sigma_min': ('FLOAT', {'default':0.0,'min':0.0,'max':1.0,'step':0.01,
                    'tooltip':'Apply the temporal bias only at noise levels >= this; later steps run native attention (motion is laid down at high noise).'}),
                'reference_token_drop': ('FLOAT', {'default':0.0,'min':0.0,'max':0.95,'step':0.05,
                    'tooltip':'Experimental (idea from CodecOPSD, arXiv 2611.31184): leave out this share of the motion video tokens that are most predictable from the previous frame (static background), so the DiT sequence is shorter. 0 = off. Needs reference_frames. With strength > 0 the exact bias runs in one pass: sage fused with per-row times (flash augmented as fallback); strength 0 keeps the selected backend.'}),
                'reference_frames': ('IMAGE', {'tooltip':'The frames of the motion video, the same ones connected to Reference to Video as <Video N>. Only needed for reference_token_drop.'})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = 'TARA-H3'
    DESCRIPTION = ('Experimental native_temporal_v1 inference for the matching motion-transfer LoRA. '
        'Keep the motion video in MiniMax H3 Reference to Video positive conditioning; route this MODEL '
        'to the guider. No Add Guide or VSA. Guide/output must share a 24fps temporal grid. '
        'Does not encode video or load the LoRA. Defaults match training.')

    def apply(self, model, motion_video_index=1, strength=2.0, window_seconds=0.2, falloff_seconds=0.25, max_penalty=6.0, attention_backend=AUTO, unique_id=None,
              structure_guidance=1.0, structure_sigma_min=0.0, strength_scale=STRENGTH_SCALES[0], bias_preset='full', bias_block_start=0, bias_block_end=99, bias_sigma_min=0.0,
              reference_token_drop=0.0, reference_frames=None):
        if BIAS_PRESETS.get(bias_preset) is not None:
            bias_block_start, bias_block_end, bias_sigma_min = BIAS_PRESETS[bias_preset]
        def report(data):
            # Optional UI telemetry must never prevent inference (including API-only use).
            try:
                from server import PromptServer
                PromptServer.instance.send_sync('tara-h3-monitor', dict(data, node=str(unique_id)))
            except Exception:
                pass
        return (patch_temporal_model(model,motion_video_index,strength,window_seconds,falloff_seconds,max_penalty,attention_backend,
                                     telemetry=report if unique_id is not None else None,
                                     structure_guidance=structure_guidance, structure_sigma_min=structure_sigma_min,
                                     bias_block_start=bias_block_start, bias_block_end=bias_block_end, bias_sigma_min=bias_sigma_min,
                                     strength_scale=strength_scale, token_drop_share=reference_token_drop,
                                     reference_frames=reference_frames),)

NODE_CLASS_MAPPINGS = {'TARAH3TemporalAlignment': TARAH3TemporalAlignment}
NODE_DISPLAY_NAME_MAPPINGS = {'TARAH3TemporalAlignment': 'TARA-H3 Temporal Alignment'}
