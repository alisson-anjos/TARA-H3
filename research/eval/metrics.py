"""Temporal-following and identity metrics for the TARA-H3 evaluation set.

Timing (appearance-independent), per frame on the 24 fps clock: head yaw/pitch/roll and mouth
opening of the largest face (insightface 3D-68) and mean Farneback flow magnitude. Each series
is correlated with the reference video's at the best lag in [-8, 8] frames.
Identity, per third of the clip: ArcFace cosine similarity of the largest face to the target
picture and to the reference-video person. Timing alone rewards copying the reference, so
read both together: good = high timing AND picture identity, low reference identity.

    python research/eval/metrics.py research/eval/clips.example.json COMFY_INPUT COMFY_OUTPUT --out metrics.csv [--prefix tara_eval]
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

KEYS = ("yaw", "pitch", "roll", "mouth", "flow")


def frames(path, n=None, size=None, start=0.0, fps=24.0):
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or fps
    out, t, i = [], start, 0
    while n is None or len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        while t <= i / src + 1e-6 and (n is None or len(out) < n):
            out.append(cv2.resize(f, size) if size else f)
            t += 1.0 / fps
        i += 1
    return out


def largest(faces):
    return max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])) if faces else None


def signals(app, fr):
    rows, embs, prev = [], [], None
    for f in fr:
        face = largest(app.get(f))
        if face is not None:
            lm = face.landmark_3d_68
            mouth = np.linalg.norm(lm[66, :2] - lm[62, :2]) / (np.linalg.norm(lm[8, :2] - lm[27, :2]) + 1e-6)
            rows.append([face.pose[1], face.pose[0], face.pose[2], mouth])
            embs.append(face.normed_embedding)
        else:
            rows.append([np.nan] * 4)
            embs.append(None)
        g = cv2.cvtColor(cv2.resize(f, (160, 160 * f.shape[0] // f.shape[1])), cv2.COLOR_BGR2GRAY)
        flow = np.nan if prev is None else np.linalg.norm(cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=2).mean()
        rows[-1].append(flow)
        prev = g
    rows = np.array(rows, dtype=float)
    return {k: rows[:, i] for i, k in enumerate(KEYS)}, embs


def best_corr(a, b, max_lag=8):
    best = (np.nan, 0)
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b[:len(b) - lag]) if lag >= 0 else (a[:lag], b[-lag:])
        m = min(len(x), len(y)); x, y = x[:m], y[:m]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 10 or np.std(x[ok]) < 1e-6 or np.std(y[ok]) < 1e-6:
            continue
        c = np.corrcoef(x[ok], y[ok])[0, 1]
        if np.isnan(best[0]) or c > best[0]:
            best = (c, lag)
    return best


def thirds(embs, target):
    n, out = len(embs), []
    for k in range(3):
        e = [x for x in embs[k * n // 3:(k + 1) * n // 3] if x is not None]
        out.append(float(np.mean(np.array(e) @ target)) if e else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("comfy_input"); ap.add_argument("comfy_output")
    ap.add_argument("--prefix", default="tara_eval")
    ap.add_argument("--out", default="metrics.csv")
    args = ap.parse_args()
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition", "landmark_3d_68"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    man = json.loads(Path(args.manifest).read_text())
    inp, outdir = Path(args.comfy_input), Path(args.comfy_output)
    handle = open(args.out, "w", newline="")
    w = csv.writer(handle)
    w.writerow(["clip", "category", "variant", "seed", *[f"{k}_r" for k in KEYS], *[f"{k}_lag" for k in KEYS],
                "timing_mean", "face_rate", "pic_1", "pic_2", "pic_3", "ref_1", "ref_2", "ref_3"])
    for clip in man["clips"]:
        picture = largest(app.get(cv2.imread(str(inp / clip.get("picture", man["picture"])))))
        size, n = (clip["width"], clip["height"]), clip["frames"]
        ref_sig, ref_embs = signals(app, frames(inp / clip["video"], n=n, size=size, start=clip.get("start", 0.0)))
        ref_person = [e for e in ref_embs if e is not None]
        ref_person = np.mean(ref_person, axis=0) / np.linalg.norm(np.mean(ref_person, axis=0)) if ref_person else None
        for vname in man["variants"]:
            for seed in man["seeds"]:
                found = sorted(outdir.glob(f"{args.prefix}/{clip['name']}/{vname}_s{seed}_*.mp4"))
                if not found:
                    continue
                sig, embs = signals(app, frames(found[-1], n=n))
                corr = [best_corr(ref_sig[k], sig[k]) for k in KEYS]
                pic = thirds(embs, picture.normed_embedding) if picture is not None else [np.nan] * 3
                ref = thirds(embs, ref_person) if ref_person is not None else [np.nan] * 3
                w.writerow([clip["name"], clip.get("category"), vname, seed, *[round(c[0], 3) for c in corr],
                            *[c[1] for c in corr], round(float(np.nanmean([c[0] for c in corr])), 3),
                            round(float(np.mean([e is not None for e in embs])), 3), *[round(x, 3) for x in pic], *[round(x, 3) for x in ref]])
                handle.flush()


if __name__ == "__main__":
    main()
