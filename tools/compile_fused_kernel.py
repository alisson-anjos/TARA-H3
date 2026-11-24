"""Ahead-of-time compile the sage fused TARA kernel for every supported GPU architecture (no GPU needed).

Usage: python tools/compile_fused_kernel.py [80 86 89 90 100 120]
"""
import sys
import types
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

sys.modules.setdefault('sageattention', types.ModuleType('sageattention'))
quant = types.ModuleType('sageattention.triton.quant_per_block')
quant.per_block_int8 = None  # runtime only; not needed to compile the attention kernel
sys.modules.setdefault('sageattention.triton', types.ModuleType('sageattention.triton'))
sys.modules.setdefault('sageattention.triton.quant_per_block', quant)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tara_h3'))
import minimax_h3_temporal_triton as fused  # noqa: E402

SIGNATURE = {
    'Q': '*i8', 'K': '*i8', 'V': '*fp16', 'Q_scale': '*fp32', 'K_scale': '*fp32', 'Out': '*bf16',
    **{name: 'i32' for name in ('stride_qh', 'stride_qn', 'stride_kh', 'stride_kn', 'stride_vh', 'stride_vn', 'stride_oh', 'stride_on',
                                'qo_len', 'kv_len', 'TA', 'TB', 'SA', 'SB', 'RT', 'RS')},
    **{name: 'fp32' for name in ('DT', 'WINDOW', 'INV_FALLOFF', 'STRENGTH', 'CAP')},
    'HEAD_DIM': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr',
}
CONSTEXPRS = {'HEAD_DIM': 128, 'BLOCK_M': 128, 'BLOCK_N': 64}

archs = [int(a) for a in sys.argv[1:]] or [80, 86, 89, 90, 100, 120]
failed = []
for arch in archs:
    try:
        src = ASTSource(fn=fused._tara_fwd, signature=SIGNATURE, constexprs=CONSTEXPRS)
        compiled = triton.compile(src, target=GPUTarget('cuda', arch, 32), options={'num_warps': 8, 'num_stages': 2})
        print(f'sm{arch}: ok (shared memory {compiled.metadata.shared} bytes)', flush=True)
    except Exception as exc:  # report every architecture before failing
        failed.append(arch)
        print(f'sm{arch}: FAILED {type(exc).__name__}: {exc}', flush=True)
sys.exit(1 if failed else 0)
