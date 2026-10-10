# Vídeo orbital sintético com poses verdadeiras, para medir as poses e o
# treino do protótipo web (ver README, "Poses no navegador").
#
#   python3 scripts/web-bench/synth-orbit.py <saida> [frames] [largura] [altura]
#
# Cena: esfera e cubo sobre um chão, com uma parede cilíndrica ao fundo, todos
# com textura procedural sem padrões repetidos; a câmera orbita com raio e
# altura variando. <saida> já é um dataset COLMAP:
#   images/*.jpg   frames 00000.jpg, 00001.jpg, …
#   sparse/0/      modelo com as poses verdadeiras (treino de referência)
#   gt.json        poses mundo→câmera e focal, para o eval-poses.py
# Requer numpy e Pillow; 120 frames 960×720 levam ~1,5 min.
import json, os, struct, sys
import numpy as np
from PIL import Image

out = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
W = int(sys.argv[3]) if len(sys.argv) > 3 else 960
H = int(sys.argv[4]) if len(sys.argv) > 4 else 720
F = 0.875 * W  # ~59° de FOV horizontal
os.makedirs(f"{out}/images", exist_ok=True)
os.makedirs(f"{out}/sparse/0", exist_ok=True)
rng = np.random.default_rng(7)
UP = np.array([0, 0, 1.0])

def make_tex(seed, palette):
    r = np.random.default_rng(seed)
    k = 48
    dirs = r.normal(size=(k, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    freqs = np.exp(r.uniform(np.log(2.0), np.log(60.0), k))
    phase = r.uniform(0, 2 * np.pi, k)
    amp = 1.0 / np.sqrt(freqs)
    chan = r.integers(0, 3, k)
    pal = np.array(palette, dtype=float)
    def tex(p):
        acc = np.zeros(p.shape[:-1] + (3,))
        for i in range(k):
            acc[..., chan[i]] += amp[i] * np.sin((p @ dirs[i]) * freqs[i] + phase[i])
        acc = 0.5 + 0.5 * np.tanh(acc * 1.2)
        return pal[0] * (1 - acc) + pal[1] * acc
    return tex

tex_sphere = make_tex(1, [[0.75, 0.15, 0.1], [0.95, 0.85, 0.3]])
tex_box = make_tex(2, [[0.1, 0.25, 0.6], [0.6, 0.9, 0.8]])
tex_ground = make_tex(3, [[0.25, 0.22, 0.18], [0.75, 0.7, 0.6]])
tex_wall = make_tex(4, [[0.2, 0.35, 0.25], [0.7, 0.75, 0.55]])
SPH_C, SPH_R = np.array([0.35, 0.0, 0.55]), 0.55
BOX_MIN, BOX_MAX = np.array([-0.95, -0.45, 0.0]), np.array([-0.25, 0.25, 0.7])
GROUND_R, WALL_R, WALL_H = 6.0, 7.0, 3.0
LIGHT = np.array([0.4, 0.3, 0.86]); LIGHT /= np.linalg.norm(LIGHT)

def shade(col, n):
    return col * (0.55 + 0.45 * np.clip((n * LIGHT).sum(-1), 0, 1))[..., None]

def render(C, D):
    shp = D.shape[:-1]
    t = np.full(shp, np.inf)
    col = np.zeros(shp + (3,))
    sky = 0.6 + 0.35 * np.clip(D[..., 2], 0, 1)
    col[:] = np.stack([sky * 0.62, sky * 0.75, sky], -1)
    def hit(ts, mask, colfn):
        nonlocal t, col
        m = mask & (ts > 1e-6) & (ts < t)
        t = np.where(m, ts, t)
        P = C + D * np.where(m, ts, 0)[..., None]
        col = np.where(m[..., None], colfn(P), col)
    # parede cilíndrica (por dentro)
    a = D[..., 0] ** 2 + D[..., 1] ** 2; b = 2 * (C[0] * D[..., 0] + C[1] * D[..., 1]); c = C[0] ** 2 + C[1] ** 2 - WALL_R ** 2
    tw = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * a + 1e-12)
    zw = C[2] + D[..., 2] * tw
    hit(tw, (zw > 0) & (zw < WALL_H), lambda P: tex_wall(P * 0.5) * 0.85)
    # chão
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = -C[2] / D[..., 2]
    tg = np.nan_to_num(tg, nan=-1, posinf=-1, neginf=-1)
    Pg = C + D * tg[..., None]
    hit(tg, np.hypot(Pg[..., 0], Pg[..., 1]) < GROUND_R, lambda P: tex_ground(P * 0.8))
    # cubo
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (BOX_MIN - C) / D; t2 = (BOX_MAX - C) / D
    tn = np.nanmax(np.minimum(t1, t2), -1); tf = np.nanmin(np.maximum(t1, t2), -1)
    def box_col(P):
        n = np.zeros_like(P)
        for ax in range(3):
            n[..., ax] = np.where(np.abs(P[..., ax] - BOX_MIN[ax]) < 1e-4, -1, 0) + np.where(np.abs(P[..., ax] - BOX_MAX[ax]) < 1e-4, 1, 0)
        return shade(tex_box(P), n)
    hit(tn, tn < tf, box_col)
    # esfera
    oc = C - SPH_C; bb = (D * oc).sum(-1); cc = oc @ oc - SPH_R ** 2; disc = bb * bb - cc
    ts = -bb - np.sqrt(np.maximum(disc, 0))
    hit(ts, disc > 0, lambda P: shade(tex_sphere(P), (P - SPH_C) / SPH_R))
    return np.clip(col, 0, 1)

def quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0] * 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q

u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
dc = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], -1)
poses = []
for i in range(N):
    a = 2 * np.pi * i / N
    h = 1.1 + 0.45 * np.sin(2 * a)
    rad = 3.2 + 0.25 * np.cos(3 * a)
    C = np.array([rad * np.cos(a), rad * np.sin(a), h])
    tgt = np.array([0, 0, 0.35])
    f = (tgt - C) / np.linalg.norm(tgt - C); r = np.cross(f, UP); r /= np.linalg.norm(r); d = np.cross(f, r)
    R = np.stack([r, d, f]); t = -R @ C
    Dw = dc @ R; Dw /= np.linalg.norm(Dw, axis=-1, keepdims=True)
    img = render(C, Dw)
    Image.fromarray((img * 255).astype(np.uint8)).save(f"{out}/images/{i:05d}.jpg", quality=92)
    poses.append({"name": f"{i:05d}.jpg", "R": R.tolist(), "t": t.tolist(), "center": C.tolist()})
json.dump({"width": W, "height": H, "focal": F, "poses": poses}, open(f"{out}/gt.json", "w"))

# Modelo COLMAP com as poses verdadeiras e pontos amostrados nas superfícies.
with open(f"{out}/sparse/0/cameras.bin", "wb") as fh:
    fh.write(struct.pack("<Q", 1)); fh.write(struct.pack("<IiQQ", 1, 0, W, H)); fh.write(struct.pack("<3d", F, W / 2, H / 2))
with open(f"{out}/sparse/0/images.bin", "wb") as fh:
    fh.write(struct.pack("<Q", N))
    for i, p in enumerate(poses):
        fh.write(struct.pack("<I4d3dI", i + 1, *quat(np.array(p["R"])), *p["t"], 1)); fh.write(p["name"].encode() + b"\0"); fh.write(struct.pack("<Q", 0))
pts = []
n = rng.normal(size=(4000, 3)); n /= np.linalg.norm(n, axis=1, keepdims=True); p = SPH_C + n * SPH_R; pts.append((p, tex_sphere(p)))
p = rng.uniform(BOX_MIN, BOX_MAX, (3000, 3)); ax = rng.integers(0, 3, 3000); side = rng.integers(0, 2, 3000)
p[np.arange(3000), ax] = np.where(side, BOX_MAX[ax], BOX_MIN[ax]); pts.append((p, tex_box(p)))
rr = GROUND_R * np.sqrt(rng.uniform(size=6000)); aa = rng.uniform(0, 2 * np.pi, 6000)
p = np.stack([rr * np.cos(aa), rr * np.sin(aa), np.zeros(6000)], 1); pts.append((p, tex_ground(p * 0.8)))
aa = rng.uniform(0, 2 * np.pi, 4000); zz = rng.uniform(0, WALL_H, 4000)
p = np.stack([WALL_R * np.cos(aa), WALL_R * np.sin(aa), zz], 1); pts.append((p, tex_wall(p * 0.5) * 0.85))
P = np.concatenate([x[0] for x in pts]); Cc = (np.clip(np.concatenate([x[1] for x in pts]), 0, 1) * 255).astype(np.uint8)
with open(f"{out}/sparse/0/points3D.bin", "wb") as fh:
    fh.write(struct.pack("<Q", len(P)))
    for i, (x, c) in enumerate(zip(P, Cc)):
        fh.write(struct.pack("<Q3d3BdQ", i + 1, *x, *c, 0.5, 0))
print(N, "frames", W, "x", H, "f =", F, "em", out)
