// pnpm test:web
import assert from "node:assert/strict";
import { test } from "node:test";
import { extractSurface, keepLargeComponents, vertexNormals } from "./extract.ts";
import { BLOCK, type Volume } from "./tsdf.ts";

/** float → IEEE half (round to nearest, enough for test values). */
function half(x: number) {
  const f = new Float32Array([x]), u = new Uint32Array(f.buffer)[0];
  const s = (u >>> 16) & 0x8000, e = ((u >>> 23) & 0xff) - 127 + 15, m = (u >>> 13) & 0x3ff;
  if (x === 0) return 0;
  if (e <= 0) return s;
  if (e >= 31) return s | 0x7c00;
  return s | (e << 10) | m;
}

/** TSDF of spheres (union) on blocks covering [lo, hi]³. */
function sphereVolume(spheres: [number, number, number, number][], voxel: number, lo: number, hi: number): Volume {
  const trunc = 3 * voxel, s = voxel * BLOCK;
  const blocks: number[] = [];
  for (let bz = Math.floor(lo / s); bz <= Math.floor(hi / s); bz++)
    for (let by = Math.floor(lo / s); by <= Math.floor(hi / s); by++)
      for (let bx = Math.floor(lo / s); bx <= Math.floor(hi / s); bx++) blocks.push(bx, by, bz);
  const nb = blocks.length / 3, BV = BLOCK ** 3;
  const tsdfw = new Uint32Array(nb * BV), color = new Uint32Array(nb * BV);
  for (let b = 0; b < nb; b++)
    for (let l = 0; l < BV; l++) {
      const g = [blocks[3 * b] * BLOCK + (l % BLOCK), blocks[3 * b + 1] * BLOCK + ((l / BLOCK) | 0) % BLOCK, blocks[3 * b + 2] * BLOCK + ((l / BLOCK ** 2) | 0)];
      const p = g.map((v) => v * voxel);
      const d = Math.min(...spheres.map(([x, y, z, r]) => Math.hypot(p[0] - x, p[1] - y, p[2] - z) - r));
      tsdfw[b * BV + l] = (half(5) << 16) | half(Math.max(-1, Math.min(1, d / trunc)));
      color[b * BV + l] = 0xff0000ff; // red, alpha 255
    }
  return { voxel, trunc, blocks: new Int32Array(blocks), tsdfw, color };
}

test("marching tetrahedra: a sphere comes out closed, outward and on its radius", () => {
  const m = extractSurface(sphereVolume([[0, 0, 0, 0.8]], 0.05, -1.2, 1.2));
  const nt = m.indices.length / 3;
  assert.ok(nt > 1000, `too few triangles: ${nt}`);
  // closed: every undirected edge in exactly two triangles, once per direction
  const dir = new Map<string, number>();
  for (let t = 0; t < nt; t++)
    for (let k = 0; k < 3; k++) {
      const a = m.indices[3 * t + k], b = m.indices[3 * t + ((k + 1) % 3)];
      dir.set(`${a},${b}`, (dir.get(`${a},${b}`) ?? 0) + 1);
    }
  for (const [e, n] of dir) {
    const [a, b] = e.split(",");
    assert.equal(n, 1, `edge ${e} repeated`);
    assert.equal(dir.get(`${b},${a}`), 1, `edge ${e} has no twin`);
  }
  // outward: positive signed volume ≈ 4/3 π r³
  let vol = 0;
  const P = m.positions;
  for (let t = 0; t < nt; t++) {
    const [a, b, c] = [0, 1, 2].map((k) => 3 * m.indices[3 * t + k]);
    vol += (P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) - P[a + 1] * (P[b] * P[c + 2] - P[b + 2] * P[c]) + P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c])) / 6;
  }
  assert.ok(Math.abs(vol / ((4 / 3) * Math.PI * 0.8 ** 3) - 1) < 0.03, `volume ${vol}`);
  for (let v = 0; v < P.length; v += 3) assert.ok(Math.abs(Math.hypot(P[v], P[v + 1], P[v + 2]) - 0.8) < 0.02);
  // colours interpolated from the voxels, normals point away from the centre
  assert.ok(Math.abs(m.colors[0] - 1) < 1e-6 && m.colors[1] === 0);
  const n = vertexNormals(m);
  for (let v = 0; v < P.length; v += 3) assert.ok(n[v] * P[v] + n[v + 1] * P[v + 1] + n[v + 2] * P[v + 2] > 0);
});

test("unobserved voxels (weight 0) leave a hole instead of a surface", () => {
  const vol = sphereVolume([[0, 0, 0, 0.8]], 0.05, -1.2, 1.2);
  for (let i = 0; i < vol.tsdfw.length; i += 2) vol.tsdfw[i] &= 0xffff; // weight 0 on half the voxels
  const full = extractSurface(sphereVolume([[0, 0, 0, 0.8]], 0.05, -1.2, 1.2)).indices.length;
  assert.ok(extractSurface(vol).indices.length < full / 4);
});

test("small components are dropped", () => {
  const m = extractSurface(sphereVolume([[0, 0, 0, 0.8], [1.05, 1.05, 1.05, 0.08]], 0.05, -1.2, 1.25));
  const kept = keepLargeComponents(m, 0.05);
  assert.ok(kept.indices.length < m.indices.length);
  for (let v = 0; v < kept.positions.length; v += 3) assert.ok(Math.hypot(kept.positions[v], kept.positions[v + 1], kept.positions[v + 2]) < 0.9);
});

test("volume → GLB: simplified to the target, standing on y = 0 with +Y up", async () => {
  const { buildGlb } = await import("./process.ts");
  // sphere centred at z = 2 with "up" = −Z: after placement the bottom is at y = 0
  const vol = sphereVolume([[0, 0, 2, 0.5]], 0.05, -0.8, 2.8);
  const { glb, stats } = await buildGlb(vol, { triangles: 500, up: [0, 0, -1] }, () => {});
  assert.ok(stats.surfaceTriangles > 2000 && stats.triangles <= 500 && stats.triangles > 300, JSON.stringify(stats));
  const dv = new DataView(glb.buffer, glb.byteOffset);
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(glb.subarray(20, 20 + jsonLen)));
  const pos = json.accessors[json.meshes[0].primitives[0].attributes.POSITION];
  assert.ok(Math.abs(pos.min[1]) < 1e-5, `floor at ${pos.min[1]}`);
  assert.ok(Math.abs(pos.max[1] - 1) < 0.03, `height ${pos.max[1]}`);
  assert.ok(Math.abs(pos.min[0] + pos.max[0]) < 1e-5 && Math.abs(pos.min[2] + pos.max[2]) < 1e-5, "centred");
});
