// Cameras and poses of the COLMAP model, for the mesh export: each image with
// its own pinhole intrinsics (lens distortion is ignored — the web SfM and the
// former desktop app's SIMPLE_PINHOLE models have none).

export interface View {
  name: string;
  /** Image size the intrinsics refer to. */
  width: number;
  height: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  /** World → camera rotation, row-major 3×3. */
  R: Float64Array;
  /** World → camera translation. */
  t: Float64Array;
  /** COLMAP quaternion (w, x, y, z) of `R`. */
  q: [number, number, number, number];
}

class Bin {
  private o = 0;
  private buf: ArrayBuffer;
  private v: DataView;
  constructor(buf: ArrayBuffer) {
    this.buf = buf;
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
  cstr() {
    const bytes = new Uint8Array(this.buf, this.o);
    const end = bytes.indexOf(0);
    const s = new TextDecoder().decode(bytes.subarray(0, end));
    this.o += end + 1;
    return s;
  }
  skip(n: number) {
    this.o += n;
  }
}

/** Parameters per COLMAP camera model id (enough to skip the record). */
const PARAMS: Record<number, number> = { 0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12 };
/** Models whose first parameter is a single focal length (f, cx, cy, …). */
const SINGLE_FOCAL = new Set([0, 2, 3, 8, 9]);

export function rotation([w, x, y, z]: readonly number[]): Float64Array {
  const n = Math.hypot(w, x, y, z) || 1;
  [w, x, y, z] = [w / n, x / n, y / n, z / n];
  return new Float64Array([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ]);
}

export function readViews(camerasBin: ArrayBuffer, imagesBin: ArrayBuffer): View[] {
  const cams = new Map<number, Omit<View, "name" | "R" | "t" | "q">>();
  const c = new Bin(camerasBin);
  for (let i = c.u64(); i > 0; i--) {
    const id = c.u32();
    const model = c.u32();
    const width = c.u64(), height = c.u64();
    const n = PARAMS[model];
    if (n == null) throw new Error(`unsupported COLMAP camera model ${model}`);
    const p = Array.from({ length: n }, () => c.f64());
    const single = SINGLE_FOCAL.has(model);
    cams.set(id, {
      width,
      height,
      fx: p[0],
      fy: single ? p[0] : p[1],
      cx: single ? p[1] : p[2],
      cy: single ? p[2] : p[3],
    });
  }
  const r = new Bin(imagesBin);
  const views: View[] = [];
  for (let i = r.u64(); i > 0; i--) {
    r.u32(); // image id
    const q: [number, number, number, number] = [r.f64(), r.f64(), r.f64(), r.f64()];
    const t = new Float64Array([r.f64(), r.f64(), r.f64()]);
    const cam = cams.get(r.u32());
    const name = r.cstr();
    r.skip(r.u64() * 24);
    if (!cam) throw new Error(`image ${name} references a missing camera`);
    views.push({ name, ...cam, R: rotation(q), t, q });
  }
  return views.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

/** Camera centre in world space: −Rᵀt. */
export function center(v: View): [number, number, number] {
  const { R, t } = v;
  return [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ];
}

/** Camera → world quaternion as [x, y, z, w] (Brush / glam order): the
 *  conjugate of COLMAP's world → camera (w, x, y, z). */
export function cameraToWorldXyzw(v: View): [number, number, number, number] {
  const [w, x, y, z] = v.q;
  const n = Math.hypot(w, x, y, z) || 1;
  return [-x / n, -y / n, -z / n, w / n];
}
