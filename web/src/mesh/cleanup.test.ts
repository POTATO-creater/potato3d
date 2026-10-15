// pnpm test:web
import assert from "node:assert/strict";
import { test } from "node:test";
import { Bvh } from "./bvh.ts";
import type { Mesh } from "./extract.ts";
import { fillHoles } from "./holes.ts";
import { smoothMesh, thinVertices } from "./smooth.ts";
import { boundaryLoops, vertexFaces, vertexNeighbours } from "./topology.ts";

/** Latitude–longitude sphere (outward), optionally without the faces above z = cut. */
function sphere(r: number, rings: number, segs: number, cut = Infinity): Mesh {
  const pos: number[] = [];
  for (let i = 0; i <= rings; i++) {
    const th = (Math.PI * i) / rings;
    for (let j = 0; j < segs; j++) {
      const ph = (2 * Math.PI * j) / segs;
      pos.push(r * Math.sin(th) * Math.cos(ph), r * Math.sin(th) * Math.sin(ph), r * Math.cos(th));
    }
  }
  const id = (i: number, j: number) => i * segs + (j % segs);
  const idx: number[] = [];
  for (let i = 0; i < rings; i++)
    for (let j = 0; j < segs; j++) {
      const q = [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)];
      for (const t of [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]) {
        if (t.some((v) => pos[3 * v + 2] > cut)) continue;
        idx.push(...t);
      }
    }
  // pole rows collapse to points: weld and drop degenerate faces
  const key = (v: number) => pos.slice(3 * v, 3 * v + 3).map((x) => Math.round(x * 1e9) || 0).join();
  const weld = new Map<string, number>(), remap: number[] = [];
  const P: number[] = [];
  for (let v = 0; v < pos.length / 3; v++) {
    const k = key(v);
    if (!weld.has(k)) weld.set(k, (P.push(pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]) - 3) / 3);
    remap.push(weld.get(k)!);
  }
  const I: number[] = [];
  for (let t = 0; t < idx.length; t += 3) {
    const [a, b, c] = [remap[idx[t]], remap[idx[t + 1]], remap[idx[t + 2]]];
    if (a !== b && b !== c && a !== c) I.push(a, b, c);
  }
  return { positions: new Float32Array(P), colors: new Float32Array(P.length).fill(0.5), indices: new Uint32Array(I) };
}

function volume(m: Mesh) {
  const P = m.positions;
  let v = 0;
  for (let t = 0; t < m.indices.length; t += 3) {
    const [a, b, c] = [0, 1, 2].map((k) => 3 * m.indices[t + k]);
    v += (P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) - P[a + 1] * (P[b] * P[c + 2] - P[b + 2] * P[c]) + P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c])) / 6;
  }
  return v;
}

test("a sphere's missing cap is closed with a dome that continues it", () => {
  const open = sphere(1, 40, 60, 0.7);
  assert.equal(boundaryLoops(open.indices, open.positions.length / 3).length, 1);
  const { mesh, stats } = fillHoles(open, { maxExtent: 10 });
  assert.equal(stats.filled, 1);
  assert.ok(stats.vertices > 20, `only ${stats.vertices} new vertices`);
  assert.equal(boundaryLoops(mesh.indices, mesh.positions.length / 3).length, 0);
  const full = volume(sphere(1, 40, 60));
  assert.ok(Math.abs(volume(mesh) / full - 1) < 0.02, `volume ${volume(mesh)} vs ${full}`);
  // a biharmonic patch is not a sphere, but close to one
  const P = mesh.positions;
  let mean = 0;
  for (let v = open.positions.length; v < P.length; v += 3) {
    const e = Math.abs(Math.hypot(P[v], P[v + 1], P[v + 2]) - 1);
    assert.ok(e < 0.08, `new vertex off the sphere by ${e}`);
    mean += e / stats.vertices;
  }
  assert.ok(mean < 0.04, `mean distance ${mean}`);
  // observed surface untouched
  assert.deepEqual(P.subarray(0, open.positions.length), open.positions);
});

test("a border wider than maxExtent is left open", () => {
  const half = sphere(1, 20, 30, 0);
  const { stats } = fillHoles(half, { maxExtent: 1 });
  assert.equal(stats.filled, 0);
});

test("smoothing flattens noise on a sphere (nothing there is thin)", () => {
  const m = sphere(1, 40, 60);
  const P = m.positions;
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
  for (let v = 0; v < P.length; v += 3) {
    const s = 1 + 0.02 * rnd();
    for (let k = 0; k < 3; k++) P[v + k] *= s;
  }
  const err = (Q: Float32Array) => {
    let e = 0;
    for (let v = 0; v < Q.length; v += 3) e += Math.abs(Math.hypot(Q[v], Q[v + 1], Q[v + 2]) - 1);
    return e / (Q.length / 3);
  };
  const { mesh, stats } = smoothMesh(m);
  assert.ok(err(mesh.positions) < 0.6 * err(P), `${err(mesh.positions)} vs ${err(P)}`);
  assert.equal(stats.thinVertices, 0);
});

test("thin parts: a narrow cylinder is thin, a large sphere is not", () => {
  const thin = (m: Mesh, t: number) => {
    const nv = m.positions.length / 3, vf = vertexFaces(m.indices, nv);
    return thinVertices(m.positions, m.indices, vertexNeighbours(m.indices, nv, vf), vf, t).reduce((s, x) => s + x, 0);
  };
  const s = sphere(1, 20, 30);
  assert.equal(thin(s, 0.5), 0);
  const c = sphere(1, 20, 30);
  for (let v = 0; v < c.positions.length; v += 3) (c.positions[v] *= 0.05), (c.positions[v + 1] *= 0.05);
  assert.ok(thin(c, 0.5) > 0.9 * (c.positions.length / 3));
});

test("BVH ray casts match brute force", () => {
  const m = sphere(1, 12, 16);
  const bvh = new Bvh(m.positions, m.indices);
  let seed = 3;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  for (let i = 0; i < 200; i++) {
    const o = [3 * rnd(), 3 * rnd(), 3 * rnd()];
    const d0 = [-o[0] + 0.5 * rnd(), -o[1] + 0.5 * rnd(), -o[2] + 0.5 * rnd()];
    const l = Math.hypot(...d0), d = d0.map((x) => x / l);
    let best = Infinity;
    for (let t = 0; t < m.indices.length / 3; t++) {
      const r = new Bvh(m.positions, m.indices.slice(3 * t, 3 * t + 3)).cast(o, d);
      best = Math.min(best, r.t);
    }
    const got = bvh.cast(o, d).t;
    assert.ok(got === best || Math.abs(got - best) < 1e-5, `${got} vs ${best}`);
  }
});

test("UV atlas: every vertex inside [0, 1], no degenerate triangle, seams duplicate vertices", async () => {
  const { unwrap } = await import("./uv.ts");
  const { vertexNormals } = await import("./extract.ts");
  const m = sphere(1, 20, 30);
  const { mesh, stats } = await unwrap(m, vertexNormals(m), 512);
  assert.ok(stats.charts >= 2, `${stats.charts} charts`);
  assert.equal(mesh.indices.length, m.indices.length);
  assert.ok(mesh.positions.length > m.positions.length);
  for (const u of mesh.uvs) assert.ok(u >= 0 && u <= 1);
  // (charts may be mirrored: the bake draws both windings)
  const U = mesh.uvs;
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const [a, b, c] = [0, 1, 2].map((k) => 2 * mesh.indices[t + k]);
    const area = (U[b] - U[a]) * (U[c + 1] - U[a + 1]) - (U[c] - U[a]) * (U[b + 1] - U[a + 1]);
    assert.ok(Math.abs(area) > 1e-9, `triangle ${t / 3} has no UV area`);
  }
});

test("textured GLB: UVs and an embedded image, no vertex colours", async () => {
  const { writeMeshGlb } = await import("./glb.ts");
  const { unwrap } = await import("./uv.ts");
  const { vertexNormals } = await import("./extract.ts");
  const m = sphere(1, 10, 12);
  const { mesh } = await unwrap(m, vertexNormals(m), 256);
  const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]);
  const glb = writeMeshGlb(mesh, [0, 0, 1], jpeg);
  const dv = new DataView(glb.buffer, glb.byteOffset);
  const json = JSON.parse(new TextDecoder().decode(glb.subarray(20, 20 + dv.getUint32(12, true))));
  const attrs = json.meshes[0].primitives[0].attributes;
  assert.ok("TEXCOORD_0" in attrs && "NORMAL" in attrs && !("COLOR_0" in attrs));
  assert.equal(json.images[0].mimeType, "image/jpeg");
  assert.equal(json.materials[0].pbrMetallicRoughness.baseColorTexture.index, 0);
});
