// Port de src-tauri/src/scene.rs: lê o modelo binário do COLMAP direto da
// pasta escolhida no navegador para enquadrar o visualizador.

import type { SceneFrame } from "../../src/lib/api";

export interface DatasetInfo {
  /** Imagens em `images/` (o Brush também aceita outros layouts). */
  images: number;
  /** Pasta do modelo COLMAP, relativa à escolhida (ex.: "sparse/0"). */
  model: string | null;
  frame: SceneFrame | null;
}

const IMAGE_EXT = /\.(jpe?g|png|webp)$/i;

async function subdir(dir: FileSystemDirectoryHandle, path: string[]) {
  let d = dir;
  for (const name of path) d = await d.getDirectoryHandle(name);
  return d;
}

async function tryFile(dir: FileSystemDirectoryHandle, name: string): Promise<ArrayBuffer | null> {
  try {
    return await (await (await dir.getFileHandle(name)).getFile()).arrayBuffer();
  } catch {
    return null;
  }
}

async function countImages(dir: FileSystemDirectoryHandle): Promise<number> {
  try {
    const images = await dir.getDirectoryHandle("images");
    let n = 0;
    for await (const [name, h] of images.entries()) if (h.kind === "file" && IMAGE_EXT.test(name)) n++;
    return n;
  } catch {
    return 0;
  }
}

export async function inspectDataset(dir: FileSystemDirectoryHandle): Promise<DatasetInfo> {
  const images = await countImages(dir);
  for (const path of [["sparse", "0"], ["sparse"], []]) {
    let model: FileSystemDirectoryHandle;
    try {
      model = await subdir(dir, path);
    } catch {
      continue;
    }
    const imagesBin = await tryFile(model, "images.bin");
    if (!imagesBin) continue;
    const pointsBin = await tryFile(model, "points3D.bin");
    let frame: SceneFrame | null = null;
    try {
      frame = readFrame(imagesBin, pointsBin);
    } catch (e) {
      console.warn("Não foi possível ler o modelo COLMAP", e);
    }
    return { images, model: path.join("/") || ".", frame };
  }
  return { images, model: null, frame: null };
}

class Bin {
  private o = 0;
  private v: DataView;
  constructor(private buf: ArrayBuffer) {
    this.v = new DataView(buf);
  }
  u32() {
    const x = this.v.getUint32(this.o, true);
    this.o += 4;
    return x;
  }
  u64() {
    const x = Number(this.v.getBigUint64(this.o, true));
    this.o += 8;
    return x;
  }
  f64() {
    const x = this.v.getFloat64(this.o, true);
    this.o += 8;
    return x;
  }
  skip(n: number) {
    this.o += n;
  }
  cstr() {
    const bytes = new Uint8Array(this.buf, this.o);
    const end = bytes.indexOf(0);
    const s = new TextDecoder().decode(bytes.subarray(0, end));
    this.o += end + 1;
    return s;
  }
}

/** Rotação mundo→câmera a partir do quatérnio (w, x, y, z) do COLMAP. */
function rotation([w, x, y, z]: number[]) {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

function median(v: number[]) {
  if (!v.length) return 0;
  v.sort((a, b) => a - b);
  return v[v.length >> 1];
}

function readFrame(imagesBin: ArrayBuffer, pointsBin: ArrayBuffer | null): SceneFrame | null {
  // images.bin: centro = -Rᵀt; "para cima" da câmera COLMAP = -Y → Rᵀ(0,-1,0).
  const r = new Bin(imagesBin);
  const n = r.u64();
  const cams: { name: string; c: [number, number, number] }[] = [];
  const up = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    r.u32();
    const q = [r.f64(), r.f64(), r.f64(), r.f64()];
    const t = [r.f64(), r.f64(), r.f64()];
    r.u32();
    const name = r.cstr();
    r.skip(r.u64() * 24); // x f64, y f64, point3D_id i64
    const m = rotation(q);
    const c: [number, number, number] = [0, 1, 2].map(
      (k) => -(m[0][k] * t[0] + m[1][k] * t[1] + m[2][k] * t[2]),
    ) as [number, number, number];
    for (let k = 0; k < 3; k++) up[k] -= m[1][k];
    cams.push({ name, c });
  }
  if (!cams.length) return null;
  cams.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  // O SplatForge grava os frames como 00000.jpg, 00001.jpg, …
  const frameIndices = cams.map(({ name }, i) => {
    const stem = name.split("/").pop()!.replace(/\.[^.]+$/, "");
    return /^\d+$/.test(stem) ? Number(stem) : i;
  });
  const cameras = cams.map((c) => c.c);
  const len = Math.max(Math.hypot(up[0], up[1], up[2]), 1e-9);

  // points3D.bin: só as coordenadas importam.
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  if (pointsBin) {
    const p = new Bin(pointsBin);
    const np = p.u64();
    for (let i = 0; i < np; i++) {
      p.u64();
      xs.push(p.f64());
      ys.push(p.f64());
      zs.push(p.f64());
      p.skip(3 + 8); // rgb, erro
      p.skip(p.u64() * 8);
    }
  }
  const center: [number, number, number] = xs.length
    ? [median(xs), median(ys), median(zs)]
    : ([0, 1, 2].map((k) => cameras.reduce((s, c) => s + c[k], 0) / cameras.length) as [number, number, number]);
  const radius = median(cameras.map((c) => Math.hypot(c[0] - center[0], c[1] - center[1], c[2] - center[2])));

  return { center, up: [up[0] / len, up[1] / len, up[2] / len], radius, cameras, frameIndices };
}
