# TARA-H3

**Temporally Aligned Reference Attention for MiniMax H3**, as ComfyUI nodes.

TARA-H3 lets the generated video follow a reference video *in time* while keeping H3's native reference conditioning. The reference video stays as native H3 reference tokens. TARA-H3 then makes two changes:

1. **Shared temporal RoPE clock.** The reference video's temporal positions are shifted onto the generated video's clock, so frame *t* of the reference and frame *t* of the output share the same time coordinate.
2. **Bounded temporal attention bias.** Generated-video queries attending to reference-video keys get a soft penalty that grows with their time distance. All other attention edges stay as they are: text, pictures, reference audio, the output's self-attention and non-target queries.

It adds no learned layers. Use it with a LoRA trained with the same policy; see [Training](#training).

The full method, equations, numerical checks and limitations are in the technical report: [research/README.md](research/README.md).

## Nodes

| Node | What it does |
|---|---|
| **TARA-H3 Temporal Alignment** | Applies the clock alignment and temporal bias to the MODEL. Route its output to the sampler/guider. |
| TARA-H3 Reference Attention Binding (Experimental) | Per-Picture logit gains on top of TARA (connect after the Temporal Alignment node). |
| TARA-H3 Token / Word Weight Editor (Experimental) | Prompt text-key logit gains. It needs traced conditioning from [BFS Native Reference Set to Video](https://github.com/alisson-anjos/ComfyUI-BFSNodes) or the H3 CLIP plus the exact prompt. |

Requirements: ComfyUI with native MiniMax H3 support; the reference video in **MiniMax H3 Reference to Video**; no Add Guide/first-frame guides on the same path; reference and output must share a 24 fps temporal grid.

## Attention backends

The bias is the same in every backend; only the kernel changes. The default picks the fastest one and falls back automatically.

| `attention_backend` | How | Needs |
|---|---|---|
| **sage fused (fastest)** (default) | One SageAttention Triton pass; the bias is computed inside the kernel, only on reference-key tiles of target-query rows | `sageattention` + `triton` |
| sage partitions | Exact log-sum-exp merge of SageAttention partitions | `sageattention` |
| flash partitions | Exact log-sum-exp merge of FlashAttention partitions | PyTorch CUDA |
| comfy kitchen | Optional one-pass CUDA extension (downloaded and verified on first use) | Linux/Windows, supported GPU |
| flex (original) | PyTorch FlexAttention score modifier | PyTorch |

Fallback chain: sage fused → sage partitions → flash partitions. With Model Sparse Attention (sol-attn) connected, the sol path is used.

## Where the bias is applied

`bias_preset` trades exactness for speed:

| Preset | Bias on | Notes |
|---|---|---|
| **full** (default) | every block, every step | matches training |
| sigma >= 0.90 (fast) | high-noise steps only | the reference timing is laid down early; later steps run native attention |
| blocks 0-11 (fast) | first 12 transformer blocks only | fastest; some reference leakage observed |
| custom | `bias_block_start`, `bias_block_end`, `bias_sigma_min` | |

## Speed

RTX 5090, MiniMax H3 Ref2VA INT8, DMD 8 steps, 416x736, 107 frames, one reference picture plus one reference video. Seconds per sampling step, same seed:

Measurements in progress; see [research/README.md](research/README.md) for the isolated attention timings.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alisson-anjos/TARA-H3
pip install sageattention triton   # optional, for the fastest backends
```

Example workflows are in [examples/](examples/). They use the TARA-H3 node IDs (`TARAH3TemporalAlignment`, ...). The same nodes also ship in ComfyUI-BFSNodes under their `BFS...` IDs for now.

## Training

TARA-H3 needs a LoRA trained with the same clock alignment and bias. The training implementation is in the ai-toolkit fork:

- Snapshot: [ai-toolkit @ tara-h3-v0.1.0](https://github.com/alisson-anjos/ai-toolkit/tree/tara-h3-v0.1.0)
- Policy and layout: [`toolkit/h3_native_temporal.py`](https://github.com/alisson-anjos/ai-toolkit/blob/tara-h3-v0.1.0/toolkit/h3_native_temporal.py)
- Example config: [`config/examples/train_h3_native_temporal.yaml`](https://github.com/alisson-anjos/ai-toolkit/blob/tara-h3-v0.1.0/config/examples/train_h3_native_temporal.yaml)
- Setup notes: [`docs/H3_NATIVE_TEMPORAL.md`](https://github.com/alisson-anjos/ai-toolkit/blob/tara-h3-v0.1.0/docs/H3_NATIVE_TEMPORAL.md)

Set `native_temporal_alignment: true`, keep `align_video_refs: false` and `align_image_refs: false`, and keep the native reference RoPE layout. Inference defaults (`strength 2.0`, `window 0.2 s`, `falloff 0.25 s`, `cap 6.0`) match the training defaults.

## Development

```bash
python -m pytest -q tests                  # unit tests (CPU)
python tools/compile_fused_kernel.py        # AOT-compile the fused Triton kernel for sm80-sm120, no GPU needed
python tools/verify_kitchen_manifest.py     # pinned Kitchen/Sol binaries exist with the pinned SHA-256
```

CI runs all three on every push and pull request. The optional Kitchen/Sol CUDA extension is built and released per commit by [alisson-anjos/comfy-kitchen](https://github.com/alisson-anjos/comfy-kitchen/tree/feat/h3-native-temporal); `tara_h3/h3_temporal_binaries.json` pins the release.

## License and citation

GPL-3.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The fused kernel is derived from [SageAttention](https://github.com/thu-ml/SageAttention) (Apache-2.0). To cite the method, use [research/CITATION.cff](research/CITATION.cff).
