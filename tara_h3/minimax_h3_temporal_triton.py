"""One-pass TARA attention: SageAttention's Triton INT8 kernel with the temporal bias fused in.

Derived from SageAttention (Apache-2.0, https://github.com/thu-ml/SageAttention,
triton/attn_qk_int8_per_block.py). The key loop is split at the motion-video rows:
keys before and after it run the unmodified Sage loop; motion keys of target-query
tiles add the TARA penalty, computed in registers from the uniform frame clock.
Same bias as the partition backends, one pass.
"""
import torch
import triton
import triton.language as tl
from sageattention.triton.quant_per_block import per_block_int8

LOG2E = 1.44269504


@triton.jit
def _inner(acc, l_i, m_i, q, q_scale, kv_len, K, V, K_scale, stride_kn, stride_vn, lo, hi,
           q_frame, q_target, SA, SB, RS, DT, WINDOW, INV_FALLOFF, STRENGTH, CAP,
           BIAS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr):
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    K_ptrs = K + (lo + offs_n[None, :]) * stride_kn + offs_k[:, None]
    V_ptrs = V + (lo + offs_n[:, None]) * stride_vn + offs_k[None, :]
    K_scale_ptr = K_scale + lo // BLOCK_N
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)
        qk += tl.where(k_mask, 0, -1.0e6)
        if BIAS:
            cols = start_n + offs_n
            k_source = (cols >= SA) & (cols < SB)
            k_frame = ((cols - SA) // RS).to(tl.float32)
            gap = tl.maximum(tl.abs(q_frame[:, None] - k_frame[None, :]) * DT - WINDOW, 0.0) * INV_FALLOFF
            penalty = tl.minimum(STRENGTH * gap * gap, CAP)
            qk -= tl.where(q_target[:, None] & k_source[None, :], penalty * 1.44269504, 0.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _tara_fwd(Q, K, V, Q_scale, K_scale, Out,
              stride_qh, stride_qn, stride_kh, stride_kn, stride_vh, stride_vn, stride_oh, stride_on,
              qo_len, kv_len, TA, TB, SA, SB, RT, RS, DT, WINDOW, INV_FALLOFF, STRENGTH, CAP,
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + off_h * stride_qh + offs_m[:, None] * stride_qn + offs_k[None, :], mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale + off_h * tl.cdiv(qo_len, BLOCK_M) + start_m)
    K = K + off_h * stride_kh
    V = V + off_h * stride_vh
    K_scale = K_scale + off_h * tl.cdiv(kv_len, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q_target = (offs_m >= TA) & (offs_m < TB)
    q_frame = ((offs_m - TA) // RT).to(tl.float32)

    lo = (SA // BLOCK_N) * BLOCK_N
    hi = tl.cdiv(SB, BLOCK_N) * BLOCK_N
    acc, l_i, m_i = _inner(acc, l_i, m_i, q, q_scale, kv_len, K, V, K_scale, stride_kn, stride_vn, 0, lo,
                           q_frame, q_target, SA, SB, RS, DT, WINDOW, INV_FALLOFF, STRENGTH, CAP,
                           False, HEAD_DIM, BLOCK_N)
    if (start_m * BLOCK_M + BLOCK_M > TA) & (start_m * BLOCK_M < TB):
        acc, l_i, m_i = _inner(acc, l_i, m_i, q, q_scale, kv_len, K, V, K_scale, stride_kn, stride_vn, lo, hi,
                               q_frame, q_target, SA, SB, RS, DT, WINDOW, INV_FALLOFF, STRENGTH, CAP,
                               True, HEAD_DIM, BLOCK_N)
    else:
        acc, l_i, m_i = _inner(acc, l_i, m_i, q, q_scale, kv_len, K, V, K_scale, stride_kn, stride_vn, lo, hi,
                               q_frame, q_target, SA, SB, RS, DT, WINDOW, INV_FALLOFF, STRENGTH, CAP,
                               False, HEAD_DIM, BLOCK_N)
    acc, l_i, m_i = _inner(acc, l_i, m_i, q, q_scale, kv_len, K, V, K_scale, stride_kn, stride_vn, hi, kv_len,
                           q_frame, q_target, SA, SB, RS, DT, WINDOW, INV_FALLOFF, STRENGTH, CAP,
                           False, HEAD_DIM, BLOCK_N)
    acc = acc / l_i[:, None]
    O_ptrs = Out + off_h * stride_oh + offs_m[:, None] * stride_on + offs_k[None, :]
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < qo_len)


def frame_step(plan):
    """Seconds between latent frames; None when the clock is not uniform (use the partition backends)."""
    t = plan.frame_times
    if len(t) < 2:
        return 0.0
    step = (t[-1] - t[0]) / (len(t) - 1)
    return step if all(abs(b - a - step) < 1e-4 for a, b in zip(t, t[1:])) else None


def attention(q, k, v, plan, strength, window, falloff, cap):
    """q, k, v: [1, heads, seq, 128] (HND). Same contract as fast.attention without reference biases."""
    if q.shape[0] != 1 or q.shape[-1] != 128 or q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError('TARA one-pass kernel needs one packed sequence with head dim 128')
    step = frame_step(plan)
    if step is None:
        raise RuntimeError('TARA one-pass kernel needs a uniform frame clock')
    k = k.contiguous()
    q_int8, q_scale, k_int8, k_scale = per_block_int8(q.contiguous(), k, km=k.mean(dim=2, keepdim=True),
                                                      sm_scale=q.shape[-1] ** -0.5, tensor_layout="HND")
    v = v.to(torch.float16).contiguous()
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    n = q.shape[2]
    grid = (triton.cdiv(n, 128), q.shape[1])
    _tara_fwd[grid](q_int8, k_int8, v, q_scale, k_scale, o,
                    q_int8.stride(1), q_int8.stride(2), k_int8.stride(1), k_int8.stride(2),
                    v.stride(1), v.stride(2), o.stride(1), o.stride(2),
                    n, n, plan.target_start, plan.target_stop, plan.source_start, plan.source_stop,
                    plan.target_frame_rows, plan.source_frame_rows,
                    float(step), float(window), 1.0 / float(falloff), float(strength), float(cap),
                    HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=2)
    return o
