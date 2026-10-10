# Compara um modelo COLMAP binário com as poses verdadeiras do synth-orbit.py:
# alinha os centros das câmeras por Sim(3) (Umeyama) e mede o erro de posição
# (em % do raio da órbita), o de rotação e o da focal.
#
#   python3 scripts/web-bench/eval-poses.py <gt.json> <pasta com cameras.bin e images.bin>
import json, struct, sys
import numpy as np

gt = json.load(open(sys.argv[1]))
model = sys.argv[2]

def read_images(p):
    out = {}
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cam = struct.unpack("<I4d3dI", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            n2 = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n2)
            w, x, y, z = qw, qx, qy, qz
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            out[name.decode()] = (R, np.array([tx, ty, tz]))
    return out

def read_focal(p):
    with open(p, "rb") as f:
        f.read(8)
        cid, model_id, w, h = struct.unpack("<IiQQ", f.read(24))
        nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 4)
        params = struct.unpack(f"<{nparams}d", f.read(8 * nparams))
        return model_id, params

est = read_images(f"{model}/images.bin")
model_id, params = read_focal(f"{model}/cameras.bin")
names = [p["name"] for p in gt["poses"] if p["name"] in est]
G = {p["name"]: p for p in gt["poses"]}
Cg = np.array([G[n]["center"] for n in names])
Ce = np.array([-est[n][0].T @ est[n][1] for n in names])

# Umeyama Sim(3): Cg ≈ s R Ce + t
mg, me = Cg.mean(0), Ce.mean(0)
X, Y = Ce - me, Cg - mg
U, S, Vt = np.linalg.svd(Y.T @ X / len(names))
D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
Rs = U @ D @ Vt
s = np.trace(np.diag(S) @ D) / (X ** 2).sum(1).mean()
ts = mg - s * Rs @ me
Ca = (s * (Rs @ Ce.T)).T + ts
radius = np.median(np.linalg.norm(Cg - Cg.mean(0), axis=1))
pos_err = np.linalg.norm(Ca - Cg, axis=1) / radius * 100

rot_err = []
for n in names:
    Rg = np.array(G[n]["R"])
    Re = est[n][0] @ Rs.T  # mundo alinhado -> câmera
    c = (np.trace(Rg.T @ Re) - 1) / 2
    rot_err.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
rot_err = np.array(rot_err)
print(f"registradas {len(names)}/{len(gt['poses'])}  focal {params[0]:.1f} (verdadeira {gt['focal']:.1f}, erro {100 * (params[0] / gt['focal'] - 1):+.2f}%)")
print(f"posição: mediana {np.median(pos_err):.3f}%  máx {pos_err.max():.3f}% do raio")
print(f"rotação: mediana {np.median(rot_err):.3f}°  máx {rot_err.max():.3f}°")
