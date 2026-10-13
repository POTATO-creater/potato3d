// Writes the model in COLMAP's binary format (cameras/images/points3D.bin),
// which Brush and the rest of Gausify Orbit read.

import { quatFromMat } from "./linalg";
import type { SfmModel } from "./reconstruct";

class Writer {
  private buf = new ArrayBuffer(1 << 16);
  private v = new DataView(this.buf);
  private o = 0;
  private ensure(n: number) {
    if (this.o + n <= this.buf.byteLength) return;
    let size = this.buf.byteLength * 2;
    while (size < this.o + n) size *= 2;
    const next = new ArrayBuffer(size);
    new Uint8Array(next).set(new Uint8Array(this.buf, 0, this.o));
    this.buf = next;
    this.v = new DataView(next);
  }
  u8(x: number) {
    this.ensure(1);
    this.v.setUint8(this.o, x);
    this.o += 1;
  }
  u32(x: number) {
    this.ensure(4);
    this.v.setUint32(this.o, x, true);
    this.o += 4;
  }
  i32(x: number) {
    this.ensure(4);
    this.v.setInt32(this.o, x, true);
    this.o += 4;
  }
  u64(x: number) {
    this.ensure(8);
    this.v.setBigUint64(this.o, BigInt(x), true);
    this.o += 8;
  }
  i64(x: number) {
    this.ensure(8);
    this.v.setBigInt64(this.o, BigInt(x), true);
    this.o += 8;
  }
  f64(x: number) {
    this.ensure(8);
    this.v.setFloat64(this.o, x, true);
    this.o += 8;
  }
  str(s: string) {
    const b = new TextEncoder().encode(s);
    this.ensure(b.length + 1);
    new Uint8Array(this.buf, this.o, b.length).set(b);
    this.o += b.length;
    this.u8(0);
  }
  blob() {
    return new Blob([this.buf.slice(0, this.o)]);
  }
}

/** `xy[i]`: keypoints of image i (pixels), for the points2D. */
export function colmapFiles(model: SfmModel, names: string[], xy: Float32Array[]): Record<string, Blob> {
  const cams = new Writer();
  cams.u64(1);
  cams.u32(1);
  cams.i32(0); // SIMPLE_PINHOLE
  cams.u64(model.width);
  cams.u64(model.height);
  cams.f64(model.f);
  cams.f64(model.cx);
  cams.f64(model.cy);

  // observations per image, for the points2D
  const perImage = new Map<number, { feat: number; pt: number }[]>();
  model.points.forEach((p, pi) => {
    for (const [i, k] of p.obs) {
      let l = perImage.get(i);
      if (!l) perImage.set(i, (l = []));
      l.push({ feat: k, pt: pi });
    }
  });
  const pt2dIndex = new Map<string, number>();
  const images = new Writer();
  const registered = model.poses.map((p, i) => [p, i] as const).filter(([p]) => p);
  images.u64(registered.length);
  for (const [pose, i] of registered) {
    const q = quatFromMat(pose!.R);
    images.u32(i + 1);
    q.forEach((v) => images.f64(v));
    pose!.t.forEach((v) => images.f64(v));
    images.u32(1);
    images.str(names[i]);
    const obs = perImage.get(i) ?? [];
    images.u64(obs.length);
    obs.forEach(({ feat, pt }, idx) => {
      images.f64(xy[i][2 * feat]);
      images.f64(xy[i][2 * feat + 1]);
      images.i64(pt + 1);
      pt2dIndex.set(`${i}:${feat}`, idx);
    });
  }

  const pts = new Writer();
  pts.u64(model.points.length);
  model.points.forEach((p, pi) => {
    pts.u64(pi + 1);
    p.X.forEach((v) => pts.f64(v));
    p.rgb.forEach((v) => pts.u8(v));
    pts.f64(p.error);
    pts.u64(p.obs.length);
    for (const [i, k] of p.obs) {
      pts.u32(i + 1);
      pts.u32(pt2dIndex.get(`${i}:${k}`) ?? 0);
    }
  });
  return { "cameras.bin": cams.blob(), "images.bin": images.blob(), "points3D.bin": pts.blob() };
}

export async function writeModel(dir: FileSystemDirectoryHandle, files: Record<string, Blob>) {
  let d = await dir.getDirectoryHandle("sparse", { create: true });
  d = await d.getDirectoryHandle("0", { create: true });
  for (const [name, blob] of Object.entries(files)) {
    const w = await (await d.getFileHandle(name, { create: true })).createWritable();
    await w.write(blob);
    await w.close();
  }
}

/** Copy of images.bin with the image names replaced (e.g. .jpg → .png of the
 *  dataset with transparency); the rest of each record is unchanged. */
export function renameImages(imagesBin: ArrayBuffer, rename: (name: string) => string): Blob {
  const v = new DataView(imagesBin);
  const bytes = new Uint8Array(imagesBin);
  const parts: BlobPart[] = [bytes.slice(0, 8)];
  let o = 8;
  for (let i = Number(v.getBigUint64(0, true)); i > 0; i--) {
    const head = o + 4 + 7 * 8 + 4; // id, qvec, tvec, camera_id
    const end = bytes.indexOf(0, head);
    parts.push(bytes.slice(o, head), new TextEncoder().encode(rename(new TextDecoder().decode(bytes.subarray(head, end)))), new Uint8Array(1));
    o = end + 1;
    const next = o + 8 + Number(v.getBigUint64(o, true)) * 24;
    parts.push(bytes.slice(o, next));
    o = next;
  }
  return new Blob(parts);
}
