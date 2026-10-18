// Reads the binary COLMAP model from OPFS to
// frame the viewer and summarize the poses.

import type { ModelStats, SceneFrame } from "../../src/lib/api";

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

/** World→camera rotation from COLMAP's (w, x, y, z) quaternion. */
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

/** COLMAP camera models with a single focal length (f, cx, cy, …); the rest
 *  start with fx, fy. */
const SINGLE_FOCAL = new Set([0, 2, 3, 8, 9]);

/** Lens of the first camera in cameras.bin. */
export function readLens(camerasBin: ArrayBuffer): SceneFrame["lens"] {
  const r = new Bin(camerasBin);
  if (!r.u64()) return undefined;
  r.u32(); // id
  const model = r.u32();
  const w = r.u64(), h = r.u64();
  const f0 = r.f64(), f1 = r.f64();
  const fy = SINGLE_FOCAL.has(model) ? f0 : f1;
  if (!(w > 0 && h > 0 && fy > 0)) return undefined;
  return { fovY: 2 * Math.atan(h / 2 / fy), aspect: w / h };
}

/** Width, height and focal lengths of the first camera in cameras.bin. */
export function readIntrinsics(camerasBin: ArrayBuffer): { width: number; height: number; fx: number; fy: number } | null {
  const r = new Bin(camerasBin);
  if (!r.u64()) return null;
  r.u32(); // id
  const model = r.u32();
  const width = r.u64(), height = r.u64();
  const f0 = r.f64(), f1 = r.f64();
  return { width, height, fx: f0, fy: SINGLE_FOCAL.has(model) ? f0 : f1 };
}

/** Poses from images.bin (world → camera), in name order. */
export function readPoses(imagesBin: ArrayBuffer): { name: string; qvec: [number, number, number, number]; tvec: [number, number, number] }[] {
  const r = new Bin(imagesBin);
  const n = r.u64();
  const out: { name: string; qvec: [number, number, number, number]; tvec: [number, number, number] }[] = [];
  for (let i = 0; i < n; i++) {
    r.u32();
    const qvec: [number, number, number, number] = [r.f64(), r.f64(), r.f64(), r.f64()];
    const tvec: [number, number, number] = [r.f64(), r.f64(), r.f64()];
    r.u32();
    const name = r.cstr();
    r.skip(r.u64() * 24);
    out.push({ name, qvec, tvec });
  }
  return out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

export function readFrame(imagesBin: ArrayBuffer, pointsBin: ArrayBuffer | null): SceneFrame | null {
  // images.bin: center = -Rᵀt; COLMAP camera "up" = -Y → Rᵀ(0,-1,0).
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
  // Gausify Orbit writes frames as 00000.jpg, 00001.jpg, …
  const frameIndices = cams.map(({ name }, i) => {
    const stem = name.split("/").pop()!.replace(/\.[^.]+$/, "");
    return /^\d+$/.test(stem) ? Number(stem) : i;
  });
  const cameras = cams.map((c) => c.c);
  const len = Math.max(Math.hypot(up[0], up[1], up[2]), 1e-9);

  // points3D.bin: only the coordinates matter.
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
      p.skip(3 + 8); // rgb, error
      p.skip(p.u64() * 8);
    }
  }
  const center: [number, number, number] = xs.length
    ? [median(xs), median(ys), median(zs)]
    : ([0, 1, 2].map((k) => cameras.reduce((s, c) => s + c[k], 0) / cameras.length) as [number, number, number]);
  const radius = median(cameras.map((c) => Math.hypot(c[0] - center[0], c[1] - center[1], c[2] - center[2])));

  return { center, up: [up[0] / len, up[1] / len, up[2] / len], radius, cameras, frameIndices };
}

/** Summary of a ready-made model (e.g. an imported dataset). */
export function modelStats(imagesBin: ArrayBuffer, pointsBin: ArrayBuffer | null, total: number, mapper: string): ModelStats {
  const registered = Number(new DataView(imagesBin).getBigUint64(0, true));
  let points = 0, err = 0;
  if (pointsBin) {
    const p = new Bin(pointsBin);
    points = p.u64();
    for (let i = 0; i < points; i++) {
      p.skip(8 + 24 + 3); // id, xyz, rgb
      err += p.f64();
      p.skip(p.u64() * 8);
    }
  }
  return { registered, total: Math.max(total, registered), points, reprojectionError: points ? err / points : null, mapper };
}
