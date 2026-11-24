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
| **auto (fastest)** (default) | Kitchen one-pass when a verified binary exists for the GPU, otherwise sage fused, then the partition backends | — |
| sage fused (triton) | One SageAttention Triton pass; the bias is computed inside the kernel, only on reference-key tiles of target-query rows | `sageattention` + `triton` |
| sage partitions | Exact log-sum-exp merge of SageAttention partitions | `sageattention` |
| flash partitions | Exact log-sum-exp merge of FlashAttention partitions | PyTorch CUDA |
| comfy kitchen | One-pass CUDA kernel from the comfy-kitchen fork (downloaded and SHA-256 verified on first use) | Linux sm80–sm120, Windows sm86/89/120 |
| flex (original) | PyTorch FlexAttention score modifier | PyTorch |

**Kitchen one-pass kernel availability.** Prebuilt binaries (built per commit by CI, SHA-256 pinned in `tara_h3/h3_temporal_binaries.json`) cover Linux sm80/86/89/90/100/120 and Windows sm86/89/120. On any other NVIDIA GPU (sm80 or newer) with a CUDA toolkit installed, TARA-H3 builds the kernel once for that GPU from source files pinned by SHA-256 at the same commit (`nvcc` from `CUDA_HOME`/`CUDA_PATH` or `PATH`; on Windows also `cl`, e.g. from an "x64 Native Tools" prompt) and caches it. Set `TARA_H3_TEMPORAL_LOCAL_BUILD=0` to skip local builds, `TARA_H3_TEMPORAL_AUTO_INSTALL=0` to skip the kernel entirely, `TARA_H3_TEMPORAL_CACHE_DIR` to move the cache. Without either, auto uses sage fused (Triton compiles it on first use).

Fallback chain for auto: Kitchen one-pass → sage fused → sage partitions → flash partitions; each failure is logged once. With **Model Sparse Attention (sol-attn)** connected, TARA uses the sol one-pass kernel with the bias fused in (and the dense chain for steps/blocks that sol leaves dense). With `strength 0` only the clock alignment remains and the selected backend runs unchanged.

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

| Attention | s/it | PSNR vs exact flash partitions |
|---|---:|---:|
| sage, no temporal bias (reference speed) | ~3.4 | — |
| flash partitions | 7.02 | reference |
| sage partitions | 5.54 | 26.5 dB |
| sage fused (triton) | 4.26 | 27.5 dB |
| **Kitchen one-pass (auto)** | **3.81** | **28.3 dB** |
| sage fused, `sigma >= 0.90` preset | 3.94 | 25.9 dB |
| sage fused, `blocks 0-11` preset | 3.58 | 17.8 dB |
| sol-attn (tau 1.3) + TARA, Kitchen one-pass | 4.93 | 25.2 dB |
| sol-attn (tau 1.3) without TARA | 3.22 | (does not follow the reference) |

Two runs of the same configuration differ by about 25–28 dB PSNR (quantized kernels are not bit-exact), so values in that range match the exact reference visually. At this short sequence (24k tokens) sol is not faster than dense; its gain grows with sequence length.

Without the TARA LoRA, the base Ref2VA model reproduces the reference video's content in this character-replacement setup, with or without TARA. TARA aligns time; the LoRA learns to use the reference video for timing while taking identity from the pictures.

## Recommendations from the measurements

- **Reference resolution matters more than output resolution.** The LoRA was trained with 672x384 reference videos (252 tokens per latent frame). At 720p with a 704x1280 reference, strength 2 lost the reference timing on some seeds; a 416x736 reference with a 720p output, or strength 0.5–1.0, tracked it without leaking the reference person's appearance. `strength_scale = auto` lowers the strength for dense references (2 → 1.07 at 704x1280); it is opt-in until evaluated on more clips.
- **Evaluation v1 (15 clips × 3 seeds, [research/eval/results](research/eval/results/)):** on short clips the bias changes neither timing nor identity; on 10 s clips it keeps identity stable (long close-up, last third: 0.69 ± 0.03 vs 0.47 ± 0.35 without the bias, which collapsed into the reference person in 1 of 3 seeds; long two-person: 0.52 vs 0.34) with equal or better timing.
- **Long clips are where the bias matters most.** On a 10 s close-up (243 frames), the run without the bias drifted into copying the reference person in the last third (ArcFace to the picture 0.07, to the reference person 0.66); with TARA identity held (0.67 / 0.08). On short or wide shots both followed timing similarly. Beyond the 4.5 s training length the output can still wander from the reference scene, so prefer clips close to the training length or windowed generation.
- `sigma >= 0.90` is a good speed preset at the training resolution, but not for dense references.

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
