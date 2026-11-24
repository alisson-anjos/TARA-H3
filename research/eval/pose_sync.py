"""Body-pose motion following: does the replaced performer move like the reference performer, at the same time?

Per frame (24 fps clock, frame 0 aligned) DWPose finds the people. The main performer is the largest
person in the reference video; in the output we take the person whose box centre is closest to it.
For each of the 17 COCO body keypoints the normalized x and y trajectories (both videos at the same
resolution) are correlated with the reference over frames where both keypoints are confident, at the
best lag in [-8, 8]. Scores: mean correlation over keypoints (motion following), its lag, and the
fraction of frames where the output performer's pose was found (turning away or disappearing lowers it).
Appearance does not enter.

    python research/eval/pose_sync.py MANIFEST COMFY_INPUT COMFY_OUTPUT --dwpose-ckpts .../comfyui_controlnet_aux/ckpts --out pose.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from metrics import frames, best_corr

BODY = 17


def load_wholebody(ckpts, controlnet_aux_src):
    sys.path.insert(0, controlnet_aux_src)
    from custom_controlnet_aux.dwpose.wholebody import Wholebody
    import torch
    c = Path(ckpts)
    det, pose = c / 'hr16/yolox-onnx/yolox_l.torchscript.pt', c / 'hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt'
    if torch.cuda.is_available() and det.is_file() and pose.is_file():
        return Wholebody(str(det), str(pose), torchscript_device='cuda')  # GPU; the ONNX path runs on CPU here
    return Wholebody(str(c / 'yzd-v/DWPose/yolox_l.onnx'), str(c / 'yzd-v/DWPose/dw-ll_ucoco_384.onnx'))


def people(model, img):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        res = model(img[:, :, ::-1].copy())  # DWPose expects RGB
    if res is None:
        return []
    kps, scores = res if isinstance(res, tuple) else (res[..., :2], res[..., 2])
    h, w = img.shape[:2]
    out = []
    for k, s in zip(kps, scores):
        k, s = np.asarray(k)[:BODY], np.asarray(s)[:BODY]
        ok = s > 0.3
        if ok.sum() < 5:
            continue
        xy = k / np.array([w, h])
        box = (xy[ok, 0].min(), xy[ok, 1].min(), xy[ok, 0].max(), xy[ok, 1].max())
        out.append((xy, ok, box))
    return out


def track(model, fr, anchor=None):
    """Per-frame keypoints of one performer: the largest (anchor None) or the one closest to anchor[t]."""
    xy = np.full((len(fr), BODY, 2), np.nan)
    centres = []
    for t, img in enumerate(fr):
        ps = people(model, img)
        if not ps:
            centres.append(None); continue
        if anchor is None or anchor[t] is None:
            p = max(ps, key=lambda p: (p[2][2] - p[2][0]) * (p[2][3] - p[2][1]))
        else:
            p = min(ps, key=lambda p: np.hypot((p[2][0] + p[2][2]) / 2 - anchor[t][0], (p[2][1] + p[2][3]) / 2 - anchor[t][1]))
        k, ok, box = p
        xy[t][ok] = k[ok]
        centres.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    return xy, centres


def pose_score(ref, out):
    rs, lags = [], []
    for j in range(BODY):
        for d in range(2):
            r, lag = best_corr(ref[:, j, d], out[:, j, d])
            if not np.isnan(r):
                rs.append(r); lags.append(lag)
    return (float(np.mean(rs)) if rs else np.nan, float(np.median(lags)) if lags else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest'); ap.add_argument('comfy_input'); ap.add_argument('comfy_output')
    ap.add_argument('--prefix', default='tara_eval'); ap.add_argument('--out', default='pose.csv')
    ap.add_argument('--dwpose-ckpts', required=True)
    ap.add_argument('--controlnet-aux-src', help='comfyui_controlnet_aux/src (default: next to the ckpts folder)')
    args = ap.parse_args()
    src = args.controlnet_aux_src or str(Path(args.dwpose_ckpts).parent / 'src')
    model = load_wholebody(args.dwpose_ckpts, src)
    man = json.loads(Path(args.manifest).read_text())
    inp, outdir = Path(args.comfy_input), Path(args.comfy_output)
    with open(args.out, 'w', newline='') as handle:
        w = csv.writer(handle)
        w.writerow(['clip', 'category', 'variant', 'seed', 'pose_r', 'pose_lag', 'pose_found', 'pose_r_1', 'pose_r_2', 'pose_r_3'])
        for clip in man['clips']:
            size, n = (clip['width'], clip['height']), clip['frames']
            ref, centres = track(model, frames(inp / clip['video'], n=n, size=size, start=clip.get('start', 0.0)))
            for vname in man['variants']:
                for seed in man['seeds']:
                    found = sorted(outdir.glob(f"{args.prefix}/{clip['name']}/{vname}_s{seed}_*.mp4"))
                    if not found:
                        continue
                    out, _ = track(model, frames(found[-1], n=n), anchor=centres)
                    r, lag = pose_score(ref, out)
                    m = len(out)
                    thirds = [pose_score(ref[k * m // 3:(k + 1) * m // 3], out[k * m // 3:(k + 1) * m // 3])[0] for k in range(3)]
                    found_rate = float(np.mean(~np.isnan(out[:, :, 0]).all(axis=1)))
                    w.writerow([clip['name'], clip.get('category'), vname, seed, round(r, 3), lag, round(found_rate, 3), *[round(x, 3) for x in thirds]])
                    handle.flush()


if __name__ == '__main__':
    main()
