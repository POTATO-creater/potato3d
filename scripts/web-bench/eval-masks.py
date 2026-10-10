# Compara as máscaras de um projeto (headless.mjs video --out) com as
# verdadeiras do synth-orbit.py --green: IoU (máscara ≥ 128) e erro médio.
#
#   python3 scripts/web-bench/eval-masks.py <masks_gt> <projeto> <fps>
import json, sys
import numpy as np
from PIL import Image

gt_dir, proj, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
ious, maes = [], []
for f in json.load(open(f"{proj}/frames.json")):
    stem = f["name"].rsplit(".", 1)[0]
    m = np.asarray(Image.open(f"{proj}/masks/{stem}.png").convert("L"), dtype=np.float32) / 255
    g = np.asarray(Image.open(f"{gt_dir}/{round(f['time'] * fps):05d}.png").convert("L"), dtype=np.float32) / 255
    if m.shape != g.shape:
        g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize(m.shape[::-1], Image.BILINEAR), dtype=np.float32) / 255
    a, b = m >= 0.5, g >= 0.5
    ious.append((a & b).sum() / max((a | b).sum(), 1))
    maes.append(np.abs(m - g).mean())
ious = np.array(ious)
print(f"{len(ious)} máscaras: IoU médio {ious.mean():.4f} (mín {ious.min():.4f}), erro médio {np.mean(maes) * 100:.2f}%")
