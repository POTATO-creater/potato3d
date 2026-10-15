// Mesh adjacency in typed arrays (compressed rows): vertex → faces, vertex →
// neighbours, face → face ring, and the boundary loops (holes).

/** items[start[i] .. start[i + 1]) belong to row i. */
export interface Rows {
  start: Uint32Array;
  items: Uint32Array;
}

/** Faces around each vertex. */
export function vertexFaces(I: Uint32Array, nv: number): Rows {
  const start = new Uint32Array(nv + 1);
  for (const v of I) start[v + 1]++;
  for (let v = 0; v < nv; v++) start[v + 1] += start[v];
  const fill = start.slice(0, nv);
  const items = new Uint32Array(I.length);
  for (let i = 0; i < I.length; i++) items[fill[I[i]]++] = (i / 3) | 0;
  return { start, items };
}

/** Unique neighbours of each vertex (through faces). */
export function vertexNeighbours(I: Uint32Array, nv: number, vf = vertexFaces(I, nv)): Rows {
  const start = new Uint32Array(nv + 1);
  const items = new Uint32Array(2 * I.length);
  let n = 0;
  for (let v = 0; v < nv; v++) {
    const s = n;
    for (let j = vf.start[v]; j < vf.start[v + 1]; j++) {
      const t = 3 * vf.items[j];
      for (let k = 0; k < 3; k++) {
        const u = I[t + k];
        if (u === v) continue;
        let seen = false;
        for (let q = s; q < n && !seen; q++) seen = items[q] === u;
        if (!seen) items[n++] = u;
      }
    }
    start[v + 1] = n;
  }
  return { start, items: items.slice(0, n) };
}

/** Faces sharing a vertex with each face (itself excluded). */
export function faceRing(I: Uint32Array, nv: number, vf = vertexFaces(I, nv)): Rows {
  const nt = I.length / 3;
  const start = new Uint32Array(nt + 1);
  let items = new Uint32Array(14 * nt);
  let n = 0;
  for (let t = 0; t < nt; t++) {
    const s = n;
    for (let k = 0; k < 3; k++) {
      const v = I[3 * t + k];
      for (let j = vf.start[v]; j < vf.start[v + 1]; j++) {
        const g = vf.items[j];
        if (g === t) continue;
        let seen = false;
        for (let q = s; q < n && !seen; q++) seen = items[q] === g;
        if (seen) continue;
        if (n === items.length) {
          const grown = new Uint32Array(items.length * 2);
          grown.set(items);
          items = grown;
        }
        items[n++] = g;
      }
    }
    start[t + 1] = n;
  }
  return { start, items: items.slice(0, n) };
}

/** Is there a face with the directed edge a → b? */
export function hasEdge(I: Uint32Array, vf: Rows, a: number, b: number) {
  for (let j = vf.start[a]; j < vf.start[a + 1]; j++) {
    const t = 3 * vf.items[j];
    for (let k = 0; k < 3; k++) if (I[t + k] === a && I[t + ((k + 1) % 3)] === b) return true;
  }
  return false;
}

/** Boundary loops, each as the vertex cycle of the hole: the hole lies on the
 *  left of v[i] → v[i+1] seen from outside, so a triangle (v[i-1], v[i],
 *  v[i+1]) has the mesh's orientation. */
export function boundaryLoops(I: Uint32Array, nv: number, vf = vertexFaces(I, nv)): number[][] {
  // hole edges b → a for every face edge a → b without a twin
  const from: number[] = [], to: number[] = [];
  for (let t = 0; t < I.length; t += 3)
    for (let k = 0; k < 3; k++) {
      const a = I[t + k], b = I[t + ((k + 1) % 3)];
      if (!hasEdge(I, vf, b, a)) from.push(b), to.push(a);
    }
  const out = new Map<number, number[]>();
  for (let e = 0; e < from.length; e++) {
    const l = out.get(from[e]);
    if (l) l.push(e);
    else out.set(from[e], [e]);
  }
  const used = new Uint8Array(from.length);
  const loops: number[][] = [];
  for (let e0 = 0; e0 < from.length; e0++) {
    if (used[e0]) continue;
    const loop: number[] = [];
    let e = e0;
    for (;;) {
      used[e] = 1;
      loop.push(from[e]);
      const v = to[e];
      if (v === from[e0]) break;
      const next = out.get(v)?.find((x) => !used[x]);
      if (next === undefined) break; // open chain (non-manifold mess): dropped below
      e = next;
    }
    if (loop.length >= 3 && to[e] === from[e0]) loops.push(loop);
  }
  return loops;
}
