#!/usr/bin/env python3
"""
LightGlue matcher for glim auto-calibration.

Takes two images (real camera + rendered LiDAR intensity), runs SuperPoint + LightGlue,
writes matches to JSON:

    { "matches": [ {"real": [u, v], "rendered": [u, v], "score": float }, ... ] }

Install once:
    pip install lightglue torch torchvision opencv-python

Or (preferred, cuda-aware):
    pip install git+https://github.com/cvg/LightGlue.git

Usage:
    python3 lightglue_match.py REAL_IMG RENDERED_IMG OUT_JSON [--max-kp N] [--min-score S]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def to_tensor(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    t = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]  # 1 x 1 x H x W
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("real", type=Path, help="real camera image (PNG/JPG)")
    ap.add_argument("rendered", type=Path, help="rendered LiDAR intensity image")
    ap.add_argument("out_json", type=Path, help="output matches JSON")
    ap.add_argument("--max-kp", type=int, default=2048)
    ap.add_argument("--min-score", type=float, default=0.2)
    ap.add_argument("--device", default=None, help="cuda | cpu (auto if omitted)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
    except ImportError as e:
        print(f"LightGlue not installed: {e}\nRun: pip install git+https://github.com/cvg/LightGlue.git",
              file=sys.stderr)
        return 2

    extractor = SuperPoint(max_num_keypoints=args.max_kp).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    t_real = to_tensor(args.real).to(device)
    t_rend = to_tensor(args.rendered).to(device)

    with torch.inference_mode():
        f_real = extractor.extract(t_real)
        f_rend = extractor.extract(t_rend)
        m = matcher({"image0": f_real, "image1": f_rend})
    f_real, f_rend, m = [rbd(x) for x in (f_real, f_rend, m)]

    kp_real = f_real["keypoints"].cpu().numpy()
    kp_rend = f_rend["keypoints"].cpu().numpy()
    matches = m["matches"].cpu().numpy()                       # (N, 2) indices
    scores = m.get("scores", None)
    if scores is not None:
        scores = scores.cpu().numpy()
    else:
        scores = np.ones(len(matches), dtype=np.float32)

    out = []
    for i, (a, b) in enumerate(matches):
        s = float(scores[i]) if i < len(scores) else 1.0
        if s < args.min_score:
            continue
        ru, rv = kp_real[a]
        qu, qv = kp_rend[b]
        out.append({
            "real": [float(ru), float(rv)],
            "rendered": [float(qu), float(qv)],
            "score": s,
        })

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump({"matches": out, "n_kp_real": len(kp_real), "n_kp_rendered": len(kp_rend)}, f)

    print(f"wrote {len(out)} matches to {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
