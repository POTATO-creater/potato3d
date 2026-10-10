// Decimação: reduz a quantidade de gaussianas fundindo as *parecidas*. Port
// de src-tauri/src/export/decimate.rs (que veio do sharp-decimate.ts do
// WebGaussian, no esquema geracional por pares do splat-transform): a cada
// geração, cada splat acha os vizinhos mais baratos de fundir (hash espacial →
// vizinhos mais próximos → custo da aresta), grupos disjuntos (pares, fechados
// em cadeia até 4 membros) são casados gulosamente em ordem de custo e cada
// grupo vira uma gaussiana por casamento de momentos.
//
// - Custo geométrico = aumento de entropia da fusão (½·logdet(Σm) − Σ pₖ·½·logdetₖ).
// - Custo de cor = |Δf_dc|².
// - Pesos e opacidade pela "tinta" (opacidade × área do elipsoide), que a
//   fusão conserva: tirar a média das opacidades abriria buracos.
// - As bandas SH também são fundidas, pela média ponderada pela tinta.

import { restLen, type Splats } from "./splats.ts";

/** Abaixo disso a decimação não compensa a perda. */
const MIN_COUNT = 10_000;
const EPS_COV = 1e-8;
const COLOR_COST_SCALE = 1;
/** Vizinhos mais próximos avaliados por splat. */
const NEIGHBOR_SCAN_CAP = 4;
/** Melhores candidatos mantidos por splat para o casamento. */
const CANDIDATES = 2;
const SELECT_BUCKETS = 1024;
/** Tamanho máximo do grupo: pares primeiro, depois fechamento em cadeia. */
const GROUP_CAPS = [3, 4];
const MAX_GENERATIONS = 24;
const MAX_GRID_RESOLUTION = 1024;
/** Entradas lidas por célula vizinha (limita células patológicas). */
const PER_CELL_CAP = 32;
const NONE = 0xffffffff;

/** Área do elipsoide (aproximação de Knud Thomsen, p ≈ 1,6075). */
function ellipsoidArea(sx: number, sy: number, sz: number) {
  const P = 1.6075;
  const a = (sx * sy) ** P, b = (sx * sz) ** P, c = (sy * sz) ** P;
  return 4 * Math.PI * ((a + b + c) / 3) ** (1 / P);
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Conjunto de trabalho de uma geração, em valores lineares. */
class Work {
  n = 0;
  readonly rl: number;
  pos: Float32Array;
  /** σ por eixo (linear) */
  scale: Float32Array;
  /** (w, x, y, z) normalizado */
  rot: Float32Array;
  /** opacidade linear (0–1) */
  opacity: Float32Array;
  dc: Float32Array;
  rest: Float32Array;
  /** Covariância no mundo, simétrica: xx, yy, zz, xy, xz, yz. */
  cov: Float64Array;
  logdet: Float64Array;
  /** opacidade × área */
  ink: Float64Array;

  constructor(capacity: number, rl: number) {
    this.rl = rl;
    this.pos = new Float32Array(capacity * 3);
    this.scale = new Float32Array(capacity * 3);
    this.rot = new Float32Array(capacity * 4);
    this.opacity = new Float32Array(capacity);
    this.dc = new Float32Array(capacity * 3);
    this.rest = new Float32Array(capacity * rl);
    this.cov = new Float64Array(capacity * 6);
    this.logdet = new Float64Array(capacity);
    this.ink = new Float64Array(capacity);
  }

  static fromSplats(s: Splats): Work {
    const rl = restLen(s);
    const w = new Work(s.count, rl);
    const rot = [0, 0, 0, 0], scale = [0, 0, 0];
    for (let i = 0; i < s.count; i++) {
      const q = s.rot.subarray(i * 4, i * 4 + 4);
      const norm = Math.max(Math.hypot(q[0], q[1], q[2], q[3]), 1e-12);
      for (let a = 0; a < 4; a++) rot[a] = q[a] / norm;
      for (let a = 0; a < 3; a++) scale[a] = Math.exp(s.scale[i * 3 + a]);
      w.push(s.pos, i * 3, scale, rot, sigmoid(s.opacity[i]), s.dc, i * 3, s.rest, i * rl);
    }
    return w;
  }

  /** Acrescenta um splat calculando covariância, logdet e tinta. */
  push(
    pos: ArrayLike<number>, po: number,
    scale: ArrayLike<number>, rot: ArrayLike<number>, opacity: number,
    dc: ArrayLike<number>, dco: number,
    rest: ArrayLike<number>, ro: number,
  ) {
    const i = this.n;
    const sx = Math.max(scale[0], 1e-9), sy = Math.max(scale[1], 1e-9), sz = Math.max(scale[2], 1e-9);
    const vx = sx * sx, vy = sy * sy, vz = sz * sz;
    const r = quatToMatrix(rot[0], rot[1], rot[2], rot[3]);
    const c = this.cov, o = i * 6;
    c[o] = r[0] * r[0] * vx + r[1] * r[1] * vy + r[2] * r[2] * vz;
    c[o + 1] = r[3] * r[3] * vx + r[4] * r[4] * vy + r[5] * r[5] * vz;
    c[o + 2] = r[6] * r[6] * vx + r[7] * r[7] * vy + r[8] * r[8] * vz;
    c[o + 3] = r[0] * r[3] * vx + r[1] * r[4] * vy + r[2] * r[5] * vz;
    c[o + 4] = r[0] * r[6] * vx + r[1] * r[7] * vy + r[2] * r[8] * vz;
    c[o + 5] = r[3] * r[6] * vx + r[4] * r[7] * vy + r[5] * r[8] * vz;
    // det(R·S²·Rᵀ) = det(S²)
    this.logdet[i] = Math.log(vx) + Math.log(vy) + Math.log(vz);
    this.ink[i] = opacity * ellipsoidArea(sx, sy, sz) + 1e-30;
    for (let a = 0; a < 3; a++) {
      this.pos[i * 3 + a] = pos[po + a];
      this.scale[i * 3 + a] = scale[a];
      this.dc[i * 3 + a] = dc[dco + a];
    }
    for (let a = 0; a < 4; a++) this.rot[i * 4 + a] = rot[a];
    this.opacity[i] = opacity;
    for (let k = 0; k < this.rl; k++) this.rest[i * this.rl + k] = rest[ro + k];
    this.n++;
  }

  copyFrom(src: Work, i: number) {
    const j = this.n, rl = this.rl;
    this.pos.set(src.pos.subarray(i * 3, i * 3 + 3), j * 3);
    this.scale.set(src.scale.subarray(i * 3, i * 3 + 3), j * 3);
    this.rot.set(src.rot.subarray(i * 4, i * 4 + 4), j * 4);
    this.opacity[j] = src.opacity[i];
    this.dc.set(src.dc.subarray(i * 3, i * 3 + 3), j * 3);
    this.rest.set(src.rest.subarray(i * rl, (i + 1) * rl), j * rl);
    this.cov.set(src.cov.subarray(i * 6, i * 6 + 6), j * 6);
    this.logdet[j] = src.logdet[i];
    this.ink[j] = src.ink[i];
    this.n++;
  }

  toSplats(shDegree: number): Splats {
    const n = this.n;
    const logit = (p: number) => {
      p = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
      return Math.log(p / (1 - p));
    };
    return {
      count: n,
      shDegree,
      pos: this.pos.slice(0, n * 3),
      scale: this.scale.slice(0, n * 3).map((s) => Math.log(Math.max(s, 1e-12))),
      rot: this.rot.slice(0, n * 4),
      opacity: this.opacity.slice(0, n).map(logit),
      dc: this.dc.slice(0, n * 3),
      rest: this.rest.slice(0, n * this.rl),
    };
  }
}

/** Funde gaussianas parecidas até restar ~`1 − reduction` delas. `reduction`
 *  vai de 0 a 0,9; zero (ou entrada pequena) devolve null. */
export function decimate(input: Splats, reduction: number, progress: (f: number) => void): Splats | null {
  reduction = Math.min(Math.max(reduction, 0), 0.9);
  if (reduction <= 0 || input.count < MIN_COUNT) return null;
  const target = Math.max(Math.round(input.count * (1 - reduction)), 1);
  let work = Work.fromSplats(input);
  const toRemove = work.n - target;
  // Uma geração travada (sem vizinhos ao alcance) amplia a busca.
  let reach = 1;
  for (let generation = 0; generation < MAX_GENERATIONS && work.n > target; ) {
    const next = runGeneration(work, target, reach);
    if (next) {
      work = next;
      generation++;
      progress(1 - (work.n - Math.min(target, work.n)) / toRemove);
    } else {
      reach *= 2;
      if (reach > 64) break;
    }
  }
  return work.n < input.count ? work.toSplats(input.shDegree) : null;
}

function runGeneration(work: Work, target: number, reach: number): Work | null {
  const { cost, idx } = buildCandidates(work, reach);
  const sel = selectMerges(work.n, cost, idx, work.n - target);
  return sel.removed > 0 ? applyMerges(work, sel) : null;
}

// ---------------------------------------------------------------------------
// Candidatos: hash espacial → vizinhos mais próximos → custo da aresta.
// ---------------------------------------------------------------------------

/** Hash de Fibonacci da chave da célula (cabe em 30 bits): os bits altos do
 *  produto, quantos a tabela usar. */
const hashSlot = (key: number, shift: number, mask: number) => (Math.imul(key, 0x9e3779b1) >>> shift) & mask;

class Grid {
  n: number[];
  min: number[];
  cell: number;
  entries: Uint32Array;
  start: Uint32Array;
  keys: Int32Array;
  ids: Uint32Array;
  mask: number;
  shift: number;
  cellCount = 0;

  constructor(work: Work, min: number[], max: number[], cell: number) {
    const n = work.n;
    this.min = min;
    this.cell = cell;
    this.n = [0, 1, 2].map((a) => Math.min(Math.max(Math.ceil((max[a] - min[a] + 1e-9) / cell), 1), MAX_GRID_RESOLUTION));
    let capacity = 2, bits = 1;
    while (capacity < n * 2) {
      capacity <<= 1;
      bits++;
    }
    this.shift = 32 - bits;
    this.keys = new Int32Array(capacity).fill(-1);
    this.ids = new Uint32Array(capacity);
    this.mask = capacity - 1;
    const cellOf = new Uint32Array(n);
    let cells = 0;
    for (let i = 0; i < n; i++) {
      const key = this.keyOf(work.pos, i * 3);
      for (let slot = hashSlot(key, this.shift, this.mask); ; slot = (slot + 1) & this.mask) {
        if (this.keys[slot] === key) {
          cellOf[i] = this.ids[slot];
          break;
        }
        if (this.keys[slot] === -1) {
          this.keys[slot] = key;
          this.ids[slot] = cells;
          cellOf[i] = cells++;
          break;
        }
      }
    }
    const start = new Uint32Array(cells + 1);
    for (let i = 0; i < n; i++) start[cellOf[i] + 1]++;
    for (let c = 0; c < cells; c++) start[c + 1] += start[c];
    const cursor = start.slice(0, cells);
    const entries = new Uint32Array(n);
    for (let i = 0; i < n; i++) entries[cursor[cellOf[i]]++] = i;
    this.entries = entries;
    this.start = start;
    this.cellCount = cells;
  }

  coord(p: ArrayLike<number>, o: number, a: number) {
    return Math.min(Math.max(Math.floor((p[o + a] - this.min[a]) / this.cell), 0), this.n[a] - 1);
  }

  key(x: number, y: number, z: number) {
    return (x * this.n[1] + y) * this.n[2] + z;
  }

  keyOf(p: ArrayLike<number>, o: number) {
    return this.key(this.coord(p, o, 0), this.coord(p, o, 1), this.coord(p, o, 2));
  }

  lookup(key: number): number {
    for (let slot = hashSlot(key, this.shift, this.mask); ; slot = (slot + 1) & this.mask) {
      const k = this.keys[slot];
      if (k === key) return this.ids[slot];
      if (k === -1) return -1;
    }
  }
}

function buildCandidates(work: Work, reach: number) {
  const n = work.n, pos = work.pos;
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) {
      const v = pos[i * 3 + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  const ext = [0, 1, 2].map((a) => Math.max(max[a] - min[a], 1e-9));

  // Célula com ocupação média baixa: o custo da varredura de 27 células é
  // ocupação × 27 por splat. Superfícies ocupam bem menos células do que o
  // palpite volumétrico supõe, daí as correções pela ocupação real.
  const minCell = Math.max(...ext) / MAX_GRID_RESOLUTION;
  let cell = Math.max(Math.cbrt((ext[0] * ext[1] * ext[2]) / n) * 2 * reach, minCell);
  let grid = new Grid(work, min, max, cell);
  for (let t = 0; t < 6; t++) {
    if (n / grid.cellCount <= 2.5 || cell <= minCell) break;
    cell = Math.max(cell / 1.6, minCell);
    grid = new Grid(work, min, max, cell);
  }

  const cost = new Float64Array(n * CANDIDATES).fill(Infinity);
  const idx = new Uint32Array(n * CANDIDATES).fill(NONE);
  const nnD = new Float64Array(NEIGHBOR_SCAN_CAP);
  const nnI = new Uint32Array(NEIGHBOR_SCAN_CAP);
  for (let i = 0; i < n; i++) {
    nnD.fill(Infinity);
    nnI.fill(NONE);
    const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
    const cx = grid.coord(pos, i * 3, 0), cy = grid.coord(pos, i * 3, 1), cz = grid.coord(pos, i * 3, 2);
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          const qx = cx + dx, qy = cy + dy, qz = cz + dz;
          if (qx < 0 || qy < 0 || qz < 0 || qx >= grid.n[0] || qy >= grid.n[1] || qz >= grid.n[2]) continue;
          const cid = grid.lookup(grid.key(qx, qy, qz));
          if (cid < 0) continue;
          const s = grid.start[cid], e = Math.min(grid.start[cid + 1], s + PER_CELL_CAP);
          for (let k = s; k < e; k++) {
            const j = grid.entries[k];
            if (j === i) continue;
            const ox = pos[j * 3] - px, oy = pos[j * 3 + 1] - py, oz = pos[j * 3 + 2] - pz;
            insertSorted(nnD, nnI, 0, NEIGHBOR_SCAN_CAP, ox * ox + oy * oy + oz * oz, j);
          }
        }
    for (let k = 0; k < NEIGHBOR_SCAN_CAP; k++)
      if (nnI[k] !== NONE) insertSorted(cost, idx, i * CANDIDATES, CANDIDATES, edgeCost(work, i, nnI[k]), nnI[k]);
  }
  return { cost, idx };
}

/** Insere `(v, j)` numa lista curta ordenada por `v`, descartando o pior. */
function insertSorted(vals: Float64Array, ids: Uint32Array, base: number, len: number, v: number, j: number) {
  const last = base + len - 1;
  // NaN também fica de fora.
  if (!(v < vals[last])) return;
  let k = last;
  while (k > base && vals[k - 1] > v) {
    vals[k] = vals[k - 1];
    ids[k] = ids[k - 1];
    k--;
  }
  vals[k] = v;
  ids[k] = j;
}

/** Custo de fundir (i, j): aumento de entropia + |Δdc|². */
function edgeCost(w: Work, i: number, j: number) {
  const wi = w.ink[i], wj = w.ink[j];
  const pi = wi / (wi + wj), pj = 1 - pi;
  const P = w.pos, C = w.cov;
  const m0 = pi * P[i * 3] + pj * P[j * 3], m1 = pi * P[i * 3 + 1] + pj * P[j * 3 + 1], m2 = pi * P[i * 3 + 2] + pj * P[j * 3 + 2];
  const di0 = P[i * 3] - m0, di1 = P[i * 3 + 1] - m1, di2 = P[i * 3 + 2] - m2;
  const dj0 = P[j * 3] - m0, dj1 = P[j * 3 + 1] - m1, dj2 = P[j * 3 + 2] - m2;
  const ci = i * 6, cj = j * 6;
  const sxx = pi * (C[ci] + di0 * di0) + pj * (C[cj] + dj0 * dj0) + EPS_COV;
  const syy = pi * (C[ci + 1] + di1 * di1) + pj * (C[cj + 1] + dj1 * dj1) + EPS_COV;
  const szz = pi * (C[ci + 2] + di2 * di2) + pj * (C[cj + 2] + dj2 * dj2) + EPS_COV;
  const sxy = pi * (C[ci + 3] + di0 * di1) + pj * (C[cj + 3] + dj0 * dj1);
  const sxz = pi * (C[ci + 4] + di0 * di2) + pj * (C[cj + 4] + dj0 * dj2);
  const syz = pi * (C[ci + 5] + di1 * di2) + pj * (C[cj + 5] + dj1 * dj2);
  const det = sxx * (syy * szz - syz * syz) - sxy * (sxy * szz - syz * sxz) + sxz * (sxy * syz - syy * sxz);
  const geo = 0.5 * (Math.log(Math.max(det, 1e-30)) - pi * w.logdet[i] - pj * w.logdet[j]);
  let color = 0;
  for (let a = 0; a < 3; a++) color += (w.dc[i * 3 + a] - w.dc[j * 3 + a]) ** 2;
  return geo + COLOR_COST_SCALE * color;
}

// ---------------------------------------------------------------------------
// Seleção: casamento guloso disjunto em ordem de custo + fechamento em cadeia.
// ---------------------------------------------------------------------------

interface Selection {
  /** Grupo de cada splat, ou NONE para os que sobrevivem. */
  groupOf: Uint32Array;
  /** Membros por grupo (CSR). */
  offsets: Uint32Array;
  members: Uint32Array;
  removed: number;
}

function selectMerges(n: number, cost: Float64Array, idx: Uint32Array, needed: number): Selection {
  let lo = Infinity, hi = -Infinity;
  for (const c of cost)
    if (Number.isFinite(c)) {
      if (c < lo) lo = c;
      if (c > hi) hi = c;
    }
  const groupOf = new Uint32Array(n).fill(NONE);
  if (!Number.isFinite(lo)) return { groupOf, offsets: new Uint32Array(1), members: new Uint32Array(0), removed: 0 };
  const span = hi > lo ? hi - lo : 1;
  const bucket = (c: number) => Math.min(Math.floor(((c - lo) / span) * SELECT_BUCKETS), SELECT_BUCKETS - 1);

  // Ordenação por contagem das entradas finitas, por faixa de custo.
  const counts = new Uint32Array(SELECT_BUCKETS + 1);
  for (const c of cost) if (Number.isFinite(c)) counts[bucket(c) + 1]++;
  for (let b = 0; b < SELECT_BUCKETS; b++) counts[b + 1] += counts[b];
  const order = new Uint32Array(counts[SELECT_BUCKETS]);
  const cursor = counts.slice(0, SELECT_BUCKETS);
  for (let e = 0; e < cost.length; e++) if (Number.isFinite(cost[e])) order[cursor[bucket(cost[e])]++] = e;

  const pairs: number[] = [];
  let removed = 0;
  // Passada principal: casa pontas livres, do mais barato para o mais caro.
  for (const e of order) {
    if (removed >= needed) break;
    const j = idx[e], i = Math.floor(e / CANDIDATES);
    if (j === NONE || groupOf[i] !== NONE || groupOf[j] !== NONE) continue;
    groupOf[i] = groupOf[j] = pairs.length / 2;
    pairs.push(i, j);
    removed++;
  }

  // Fechamento em cadeia: splats ainda livres entram no grupo de um
  // candidato, para alvos altos fecharem em menos gerações.
  const groups = pairs.length / 2;
  const size = new Uint32Array(groups).fill(2);
  const joins: number[] = [];
  for (const cap of GROUP_CAPS)
    for (const e of order) {
      if (removed >= needed) break;
      const j = idx[e], i = Math.floor(e / CANDIDATES);
      if (j === NONE || groupOf[i] !== NONE) continue;
      const g = groupOf[j];
      if (g === NONE || size[g] >= cap) continue;
      groupOf[i] = g;
      size[g]++;
      joins.push(i, g);
      removed++;
    }

  const offsets = new Uint32Array(groups + 1);
  for (let g = 0; g < groups; g++) offsets[g + 1] = offsets[g] + size[g];
  const members = new Uint32Array(offsets[groups]);
  const fill = new Uint32Array(groups).fill(2);
  for (let g = 0; g < groups; g++) {
    members[offsets[g]] = pairs[2 * g];
    members[offsets[g] + 1] = pairs[2 * g + 1];
  }
  for (let k = 0; k < joins.length; k += 2) {
    const g = joins[k + 1];
    members[offsets[g] + fill[g]++] = joins[k];
  }
  return { groupOf, offsets, members, removed };
}

// ---------------------------------------------------------------------------
// Fusão: casamento de momentos n-ário que conserva a tinta.
// ---------------------------------------------------------------------------

function applyMerges(w: Work, sel: Selection): Work {
  const out = new Work(w.n - sel.removed, w.rl);
  const emitted = new Uint8Array(sel.offsets.length - 1);
  const rest = new Float64Array(w.rl);
  for (let i = 0; i < w.n; i++) {
    const g = sel.groupOf[i];
    if (g === NONE) out.copyFrom(w, i);
    else if (!emitted[g]) {
      emitted[g] = 1;
      mergeGroup(w, sel.members.subarray(sel.offsets[g], sel.offsets[g + 1]), out, rest);
    }
  }
  return out;
}

function mergeGroup(w: Work, members: Uint32Array, out: Work, rest: Float64Array) {
  let total = 0;
  for (const i of members) total += w.ink[i];
  const m = [0, 0, 0];
  for (const i of members) for (let a = 0; a < 3; a++) m[a] += (w.ink[i] / total) * w.pos[i * 3 + a];
  const s = [EPS_COV, EPS_COV, EPS_COV, 0, 0, 0];
  const dc = [0, 0, 0];
  rest.fill(0);
  for (const i of members) {
    const p = w.ink[i] / total;
    const d0 = w.pos[i * 3] - m[0], d1 = w.pos[i * 3 + 1] - m[1], d2 = w.pos[i * 3 + 2] - m[2];
    const c = i * 6, C = w.cov;
    s[0] += p * (C[c] + d0 * d0);
    s[1] += p * (C[c + 1] + d1 * d1);
    s[2] += p * (C[c + 2] + d2 * d2);
    s[3] += p * (C[c + 3] + d0 * d1);
    s[4] += p * (C[c + 4] + d0 * d2);
    s[5] += p * (C[c + 5] + d1 * d2);
    for (let a = 0; a < 3; a++) dc[a] += p * w.dc[i * 3 + a];
    for (let k = 0; k < w.rl; k++) rest[k] += p * w.rest[i * w.rl + k];
  }

  const mat = [s[0], s[3], s[4], s[3], s[1], s[5], s[4], s[5], s[2]];
  const vecs = jacobiEigen(mat);
  const ev = [mat[0], mat[4], mat[8]];
  sortEigenDesc(ev, vecs);
  ensureProperRotation(vecs);
  const sc = ev.map((v) => Math.sqrt(Math.max(v, 1e-12)));
  const q = quatFromMatrix(vecs);

  // Opacidade que conserva a tinta: a tinta do grupo espalhada pela área do
  // splat fundido, limitada a 1.
  const area = Math.max(ellipsoidArea(sc[0], sc[1], sc[2]), 1e-30);
  out.push(m, 0, sc, q, Math.min(total / area, 1), dc, 0, rest, 0);
}

// ---------------------------------------------------------------------------
// Álgebra 3×3 (linha maior), igual ao Rust.
// ---------------------------------------------------------------------------

/** Matriz de rotação a partir de (w, x, y, z) normalizado. */
export function quatToMatrix(w: number, x: number, y: number, z: number): number[] {
  return [
    w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z,
  ];
}

/** Jacobi no lugar: a diagonal de `m` vira os autovalores; as colunas do
 *  retorno são os autovetores. */
function jacobiEigen(m: number[]): number[] {
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let it = 0; it < 8; it++) {
    if (Math.abs(m[1]) + Math.abs(m[2]) + Math.abs(m[5]) < 1e-14) break;
    jacobiRotate(m, v, 0, 1);
    jacobiRotate(m, v, 0, 2);
    jacobiRotate(m, v, 1, 2);
  }
  return v;
}

function jacobiRotate(m: number[], v: number[], p: number, q: number) {
  const pp = p * 3 + p, qq = q * 3 + q, pq = p * 3 + q, qp = q * 3 + p;
  const app = m[pp], aqq = m[qq], apq = m[pq];
  if (Math.abs(apq) < 1e-18) return;
  const tau = (aqq - app) / (2 * apq);
  const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
  const c = 1 / Math.sqrt(1 + t * t), s = t * c;
  for (let k = 0; k < 3; k++) {
    if (k === p || k === q) continue;
    const mkp = m[k * 3 + p], mkq = m[k * 3 + q];
    const nkp = c * mkp - s * mkq, nkq = s * mkp + c * mkq;
    m[k * 3 + p] = m[p * 3 + k] = nkp;
    m[k * 3 + q] = m[q * 3 + k] = nkq;
  }
  m[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
  m[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
  m[pq] = m[qp] = 0;
  for (let k = 0; k < 3; k++) {
    const vkp = v[k * 3 + p], vkq = v[k * 3 + q];
    v[k * 3 + p] = c * vkp - s * vkq;
    v[k * 3 + q] = s * vkp + c * vkq;
  }
}

function sortEigenDesc(ev: number[], v: number[]) {
  for (const [a, b] of [[0, 1], [1, 2], [0, 1]]) {
    if (ev[a] < ev[b]) {
      [ev[a], ev[b]] = [ev[b], ev[a]];
      for (let row = 0; row < 3; row++) [v[row * 3 + a], v[row * 3 + b]] = [v[row * 3 + b], v[row * 3 + a]];
    }
  }
}

/** Inverte uma coluna se preciso para a base ser rotação própria (det = +1). */
function ensureProperRotation(r: number[]) {
  const det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
  if (det < 0) {
    r[2] = -r[2];
    r[5] = -r[5];
    r[8] = -r[8];
  }
}

export function quatFromMatrix(r: number[]): number[] {
  const [r00, r01, r02, r10, r11, r12, r20, r21, r22] = r;
  const trace = r00 + r11 + r22;
  let q: number[];
  if (trace > 0) {
    const s = 2 * Math.sqrt(Math.max(trace + 1, 1e-12));
    q = [0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s];
  } else if (r00 > r11 && r00 > r22) {
    const s = 2 * Math.sqrt(Math.max(1 + r00 - r11 - r22, 1e-12));
    q = [(r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s];
  } else if (r11 > r22) {
    const s = 2 * Math.sqrt(Math.max(1 + r11 - r00 - r22, 1e-12));
    q = [(r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s];
  } else {
    const s = 2 * Math.sqrt(Math.max(1 + r22 - r00 - r11, 1e-12));
    q = [(r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s];
  }
  const norm = Math.max(Math.hypot(...q), 1e-12);
  return q.map((v) => v / norm);
}

export { ellipsoidArea, sigmoid };
