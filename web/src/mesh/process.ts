// TSDF volume → mesh: surface extraction, cleanup, hole filling,
// simplification, feature-preserving smoothing and (for a texture) the UV
// atlas; runs in mesh.worker.ts (pure, also tested in Node). Then, after the
// texture bake on the main thread, y-up placement and the binary glTF.

import { MeshoptSimplifier } from "meshoptimizer";
import type { Msg } from "../../../src/lib/api";
import { extractSurface, keepLargeComponents, type Mesh, vertexNormals } from "./extract.ts";
import { writeMeshGlb } from "./glb.ts";
import { fillHoles } from "./holes.ts";
import { smoothMesh, uniformEdge } from "./smooth.ts";
import type { Volume } from "./tsdf.ts";
import { unwrap } from "./uv.ts";

export interface BuildOptions {
  /** Target triangle count after simplification. */
  triangles: number;
  /** World "up" (unit vector); the GLB is rotated so it becomes +Y. */
  up: [number, number, number];
  /** Texture size for the UV atlas; 0 = vertex colours only. */
  texture: number;
}

/** World-space mesh with normals, and UVs when a texture was asked for. */
export interface BuiltMesh extends Mesh {
  normals: Float32Array;
  uvs?: Float32Array;
}

export interface BuildStats {
  /** Triangles straight out of the TSDF. */
  surfaceTriangles: number;
  triangles: number;
  vertices: number;
  /** Boundary loops found / closed. */
  holes: number;
  holesFilled: number;
  /** Faces folded against their neighbourhood, before and after smoothing. */
  folds: [number, number];
  /** UV charts (0 without a texture). */
  charts: number;
}

/** Drops unreferenced vertices (after simplification). */
export function compact(m: Mesh, indices: Uint32Array): Mesh {
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

/** Bounding-box diagonal. */
function diagonal(P: Float32Array) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < P.length; v += 3)
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], P[v + k]);
      hi[k] = Math.max(hi[k], P[v + k]);
    }
  return Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
}

export async function buildMesh(
  vol: Volume,
  o: BuildOptions,
  progress: (fraction: number, message: Msg) => void,
): Promise<{ mesh: BuiltMesh; stats: BuildStats }> {
  progress(0, { key: "mesh.extracting" });
  let m = extractSurface(vol);
  const surfaceTriangles = m.indices.length / 3;
  if (!surfaceTriangles) throw new Error("no surface found: the depth maps did not agree on any part of the subject");
  m = keepLargeComponents(m, 0.01);

  // Holes (what no photo saw) closed at about the final edge length: finer
  // patches would only be simplified away.
  progress(0.15, { key: "mesh.filling" });
  const finalEdge = uniformEdge(m.positions, m.indices) * Math.sqrt(m.indices.length / 3 / o.triangles);
  const fill = fillHoles(m, { maxExtent: 0.5 * diagonal(m.positions), edge: 0.5 * finalEdge });
  m = fill.mesh;

  progress(0.3, { key: "mesh.simplifying" });
  await MeshoptSimplifier.ready;
  if (m.indices.length / 3 > o.triangles) {
    // The count governs: allow any error, the target stops it. Regularize
    // trades a little detail for even triangles, which smooth (and unwrap)
    // with far fewer folds.
    const [indices] = MeshoptSimplifier.simplify(m.indices, m.positions, 3, o.triangles * 3, 1, ["Regularize"]);
    m = compact(m, indices);
  }

  progress(0.45, { key: "mesh.smoothing" });
  const smooth = smoothMesh(m);
  m = smooth.mesh;
  let mesh: BuiltMesh = { ...m, normals: vertexNormals(m) };
  let charts = 0;
  if (o.texture) {
    progress(0.6, { key: "mesh.unwrapping" });
    const uv = await unwrap(m, mesh.normals, o.texture);
    mesh = uv.mesh;
    charts = uv.stats.charts;
  }
  progress(1, { key: "mesh.unwrapping" });
  return {
    mesh,
    stats: {
      surfaceTriangles,
      triangles: mesh.indices.length / 3,
      vertices: mesh.positions.length / 3,
      holes: fill.stats.holes,
      holesFilled: fill.stats.filled,
      folds: [smooth.stats.foldsBefore, smooth.stats.foldsAfter],
      charts,
    },
  };
}

/** Volume → GLB with vertex colours (no texture). */
export async function buildGlb(
  vol: Volume,
  o: Omit<BuildOptions, "texture">,
  progress: (fraction: number, message: Msg) => void,
): Promise<{ glb: Uint8Array; stats: BuildStats }> {
  const { mesh, stats } = await buildMesh(vol, { ...o, texture: 0 }, progress);
  return { glb: writeMeshGlb(mesh, o.up), stats };
}
