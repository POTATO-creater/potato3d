// TSDF volume → GLB: surface extraction, cleanup, simplification, y-up
// placement and the binary glTF. Runs in mesh.worker.ts (pure, also tested in
// Node).

import { MeshoptSimplifier } from "meshoptimizer";
import type { Msg } from "../../../src/lib/api";
import { extractSurface, keepLargeComponents, type Mesh, vertexNormals } from "./extract.ts";
import { srgbToLinear, writeGlb } from "./glb.ts";
import type { Volume } from "./tsdf.ts";

export interface BuildOptions {
  /** Target triangle count after simplification. */
  triangles: number;
  /** World "up" (unit vector); the GLB is rotated so it becomes +Y. */
  up: [number, number, number];
}

export interface BuildStats {
  /** Triangles straight out of the TSDF. */
  surfaceTriangles: number;
  triangles: number;
  vertices: number;
}

/** Drops unreferenced vertices (after simplification). */
function compact(m: Mesh, indices: Uint32Array): Mesh {
  const remap = new Int32Array(m.positions.length / 3).fill(-1);
  let n = 0;
  for (const v of indices) if (remap[v] < 0) remap[v] = n++;
  const positions = new Float32Array(3 * n), colors = new Float32Array(3 * n);
  for (let v = 0; v < remap.length; v++) {
    const r = remap[v];
    if (r < 0) continue;
    positions.set(m.positions.subarray(3 * v, 3 * v + 3), 3 * r);
    colors.set(m.colors.subarray(3 * v, 3 * v + 3), 3 * r);
  }
  return { positions, colors, indices: indices.map((v) => remap[v]) };
}

/** Rotation taking `up` to +Y (proper, right-handed), as rows. */
function yUp(up: [number, number, number]) {
  const u = up.map((x) => x / (Math.hypot(...up) || 1)) as [number, number, number];
  // any axis not parallel to up, made orthogonal
  const a: [number, number, number] = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
  const d = a[0] * u[0] + a[1] * u[1] + a[2] * u[2];
  const x = a.map((v, i) => v - d * u[i]);
  const xl = Math.hypot(...x);
  const xr = x.map((v) => v / xl) as [number, number, number];
  // z = x × y keeps it right-handed
  const z: [number, number, number] = [xr[1] * u[2] - xr[2] * u[1], xr[2] * u[0] - xr[0] * u[2], xr[0] * u[1] - xr[1] * u[0]];
  return [xr, u, z];
}

export async function buildGlb(
  vol: Volume,
  o: BuildOptions,
  progress: (fraction: number, message: Msg) => void,
): Promise<{ glb: Uint8Array; stats: BuildStats }> {
  progress(0, { key: "mesh.extracting" });
  let m = extractSurface(vol);
  const surfaceTriangles = m.indices.length / 3;
  if (!surfaceTriangles) throw new Error("no surface found: the depth maps did not agree on any part of the subject");
  m = keepLargeComponents(m, 0.01);

  progress(0.4, { key: "mesh.simplifying" });
  await MeshoptSimplifier.ready;
  if (m.indices.length / 3 > o.triangles) {
    // The count governs: allow any error, the target stops it.
    const [indices] = MeshoptSimplifier.simplify(m.indices, m.positions, 3, o.triangles * 3, 1, []);
    m = compact(m, indices);
  }

  progress(0.8, { key: "mesh.writing" });
  // y-up, centred on the vertical axis, standing on y = 0.
  const R = yUp(o.up);
  const P = m.positions;
  const out = new Float32Array(P.length);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < P.length; v += 3) {
    for (let r = 0; r < 3; r++) {
      const x = R[r][0] * P[v] + R[r][1] * P[v + 1] + R[r][2] * P[v + 2];
      out[v + r] = x;
      lo[r] = Math.min(lo[r], x);
      hi[r] = Math.max(hi[r], x);
    }
  }
  const shift = [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2];
  for (let v = 0; v < out.length; v += 3) for (let r = 0; r < 3; r++) out[v + r] -= shift[r];
  const placed: Mesh = { ...m, positions: out };
  const colors = m.colors.map(srgbToLinear);
  const glb = writeGlb({ positions: out, normals: vertexNormals(placed), colors, indices: m.indices });
  progress(1, { key: "mesh.writing" });
  return { glb, stats: { surfaceTriangles, triangles: m.indices.length / 3, vertices: out.length / 3 } };
}
