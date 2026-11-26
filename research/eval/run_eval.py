"""Generate the TARA-H3 evaluation set through a running ComfyUI (API).

Every clip in the manifest is rendered with each variant and seed from the same
inputs, so variants differ only in the temporal policy. Videos and the reference
clip must be in ComfyUI's input folder; outputs go to output/<prefix>/<clip>/.

    python research/eval/run_eval.py research/eval/clips.example.json --url http://127.0.0.1:8188

Manifest: {"picture": "face.png", "prompt_file": "prompt.txt",
           "models": {...optional overrides...},
           "clips": [{"name": "closeup_01", "video": "clip.mp4", "category": "closeup",
                      "width": 416, "height": 736, "frames": 107, "start": 0.0}],
           "variants": {"native": {"native": true}, "clock_only": {"strength": 0.0}, "tara": {"strength": 2.0}},
           "seeds": [42, 7]}
Frame counts must be 17n+5 (90, 107, 124, 243, ...).
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

MODELS = {
    "unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "turbo_lora": "dmd_ref2va_8step.safetensors",
    "tara_lora": "minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "steps": 8,
}


def graph(clip, variant, seed, picture, prompt, models, prefix):
    tara = {"model": ["4", 0], "motion_video_index": 1, "strength": 2.0, "window_seconds": 0.2,
            "falloff_seconds": 0.25, "max_penalty": 6.0, "attention_backend": "auto (fastest)"}
    tara.update({k: v for k, v in variant.items() if k not in ("tara_lora_strength", "reference_width", "reference_height", "native")})
    W, H, n = clip["width"], clip["height"], clip["frames"]
    rw, rh = variant.get("reference_width", W), variant.get("reference_height", H)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": models["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": models["turbo_lora"], "strength_model": 1.0, "model": ["1", 0]}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": models["tara_lora"],
              "strength_model": variant.get("tara_lora_strength", 1.0), "model": ["2", 0]}},
        "4": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"shift_video": 12.0, "shift_audio": 3.0, "model": ["3", 0]}},
        "5": {"class_type": "TARAH3TemporalAlignment", "inputs": tara},
        "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": models["clip"], "type": "minimax", "device": "default"}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": models["audio_vae"]}},
        "10": {"class_type": "VHS_LoadVideoFFmpeg", "inputs": {"video": clip["video"], "force_rate": 24.0, "custom_width": rw,
               "custom_height": rh, "frame_load_cap": n, "start_time": clip.get("start", 0.0)}},
        "11": {"class_type": "LoadImage", "inputs": {"image": clip.get("picture", picture)}},
        "12": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["6", 0], "vae": ["7", 0], "audio_vae": ["8", 0],
               "prompt": prompt, "width": W, "height": H, "length": n, "ref_image_size": "max",
               "ref_images.ref_image_0": ["11", 0], "ref_videos.ref_video_0": ["10", 0]}},
        "13": {"class_type": "CFGGuider", "inputs": {"cfg": 1.0, "model": ["5", 0], "positive": ["12", 0], "negative": ["12", 0]}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "16": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": models["steps"], "denoise": 1.0, "model": ["5", 0]}},
        "17": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["14", 0], "guider": ["13", 0], "sampler": ["15", 0],
               "sigmas": ["16", 0], "latent_image": ["12", 1]}},
        "18": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["17", 0]}},
        "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["7", 0]}},
        "20": {"class_type": "CreateVideo", "inputs": {"fps": 24.0, "images": ["19", 0]}},
        "21": {"class_type": "SaveVideo", "inputs": {"filename_prefix": prefix, "format": "auto", "codec": "auto", "video": ["20", 0]}},
    }
    if variant.get("reference_token_drop", 0) > 0:
        # Codec-style pruning of the motion video's tokens needs the same frames that go into Reference to Video.
        g["5"]["inputs"]["reference_frames"] = ["10", 0]
    if variant.get("native"):
        # Baseline: MiniMax H3 native reference conditioning, no TARA node (no clock alignment, no bias).
        del g["5"]
        g["13"]["inputs"]["model"] = ["4", 0]
        g["16"]["inputs"]["model"] = ["4", 0]
    return g


def call(url, path, data=None):
    req = urllib.request.Request(url + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--prefix", default="tara_eval")
    ap.add_argument("--output-dir", help="ComfyUI output folder, to skip renders that already exist")
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text())
    base = Path(args.manifest).parent
    prompt = (base / man["prompt_file"]).read_text()
    models = dict(MODELS, **man.get("models", {}))
    log = Path(args.prefix + "_runs.jsonl")
    for clip in man["clips"]:
        for vname, variant in man["variants"].items():
            for seed in man["seeds"]:
                prefix = f"{args.prefix}/{clip['name']}/{vname}_s{seed}"
                if args.output_dir and list(Path(args.output_dir).glob(prefix + "_*.mp4")):
                    continue
                t = time.time()
                pid = call(args.url, "/prompt", {"prompt": graph(clip, variant, seed, man["picture"], prompt, models, prefix)})["prompt_id"]
                while True:
                    h = call(args.url, f"/history/{pid}")
                    if pid in h and h[pid].get("status", {}).get("completed") is not None:
                        break
                    time.sleep(3)
                rec = {"clip": clip["name"], "category": clip.get("category"), "variant": vname, "seed": seed,
                       "status": h[pid]["status"].get("status_str"), "seconds": round(time.time() - t, 1)}
                print(json.dumps(rec), flush=True)
                with log.open("a") as f:
                    f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
