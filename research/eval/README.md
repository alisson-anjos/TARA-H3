# TARA-H3 evaluation

Reproducible version of the measurements in the technical report (Sections 4.4–4.6).

1. Put the reference videos and the identity picture in ComfyUI's `input/` folder and start ComfyUI with TARA-H3 installed.
2. Write a manifest like [`clips.example.json`](clips.example.json): clips (category, size, 17n+5 frame count, start time), variants (any `TARAH3TemporalAlignment` inputs, plus `tara_lora_strength`, `reference_width`, `reference_height`) and seeds. Model file names can be overridden under `"models"`.
3. Render, measure, summarize:

```bash
python research/eval/run_eval.py my_clips.json --url http://127.0.0.1:8188 --output-dir ComfyUI/output
python research/eval/metrics.py my_clips.json ComfyUI/input ComfyUI/output --out metrics.csv   # needs insightface + opencv
python research/eval/summarize.py metrics.csv > results.md
```

Variants: `{"native": true}` renders MiniMax H3's native reference conditioning without the TARA node (the baseline for temporal alignment); `{"strength": 0.0}` keeps only TARA's clock alignment; `{"strength": 2.0}` is full TARA. Add `"reference_token_drop": 0.5` to any TARA variant to drop that share of the motion video's tokens (the harness connects the reference frames).

TARA's purpose is temporal alignment, so the primary measures are motion-following ones: `pose_sync.py` (DWPose body keypoints of the main performer, for medium/wide/full-body shots) and the face-motion columns of `metrics.py` (head yaw/pitch/roll and mouth, for close-ups):

```bash
python research/eval/pose_sync.py my_clips.json ComfyUI/input ComfyUI/output --dwpose-ckpts ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts --out pose.csv
```

`metrics.py` reports, per render: timing correlation with the reference video (head yaw/pitch/roll, mouth opening, optical-flow magnitude; best lag within ±8 frames) and ArcFace identity per third of the clip, against the target picture (`pic_*`) and the reference-video person (`ref_*`). Timing alone rewards copying the reference, so read it together with identity; `collapse` marks renders whose last third looks more like the reference person than the picture.
