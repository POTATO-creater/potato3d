// UV atlas with xatlas: the mesh is cut into charts that unfold with little
// stretch, packed into one square texture. Vertices on chart seams are
// duplicated; their normals and colours stay those of the welded mesh, so the
// shading has no seams.

import createXAtlas from "xatlas-wasm";
import type { Mesh } from "./extract.ts";

export interface UvMesh extends Mesh {
  normals: Float32Array;
  /** 0–1, v down (glTF). */
  uvs: Float32Array;
}

export interface UvStats {
  charts: number;
  /** Share of the texture covered by charts. */
  utilization: number;
}

export async function unwrap(m: Mesh, normals: Float32Array, size: number): Promise<{ mesh: UvMesh; stats: UvStats }> {
  const xatlas = await createXAtlas();
  const atlas = xatlas.createAtlas();
  try {
    const err = atlas.addMesh({ positions: m.positions, normals, indices: m.indices });
    if (err !== xatlas.AddMeshError.Success) throw new Error(`UV unwrap: ${xatlas.addMeshErrorString(err)}`);
    // padding in texels, so bilinear and mip sampling of one chart don't reach the next
    atlas.generate({}, { resolution: size, padding: Math.max(2, Math.round(size / 1024)), bilinear: true });
    const out = atlas.getMesh(0);
    const n = out.vertexCount, w = atlas.width, h = atlas.height;
    const positions = new Float32Array(3 * n), norms = new Float32Array(3 * n), colors = new Float32Array(3 * n);
    const uvs = new Float32Array(2 * n);
    out.vertices.forEach((v, i) => {
      const s = 3 * v.xref;
      positions.set(m.positions.subarray(s, s + 3), 3 * i);
      norms.set(normals.subarray(s, s + 3), 3 * i);
      colors.set(m.colors.subarray(s, s + 3), 3 * i);
      uvs[2 * i] = v.uv[0] / w;
      uvs[2 * i + 1] = v.uv[1] / h;
    });
    return {
      mesh: { positions, normals: norms, colors, uvs, indices: Uint32Array.from(out.indices) },
      stats: { charts: atlas.chartCount, utilization: atlas.getUtilization(0) },
    };
  } finally {
    atlas.destroy();
  }
}
