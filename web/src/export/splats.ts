// Leitura e escrita do .ply de Gaussian Splatting: port de
// src-tauri/src/export/ply.rs. O Brush grava as propriedades numa ordem
// própria (f_dc, f_rest, opacity, rot, scale, xyz), então a leitura localiza
// cada coluna pelo nome; a escrita usa a ordem do INRIA, que todo
// visualizador entende.

/** Gaussianas em colunas, com os valores crus do .ply: escala em log,
 *  opacidade em logit e cor como coeficientes SH. */
export interface Splats {
  count: number;
  /** Grau SH das bandas `f_rest` (0–3). */
  shDegree: number;
  /** x, y, z */
  pos: Float32Array;
  /** log(σ) por eixo */
  scale: Float32Array;
  /** (w, x, y, z), não necessariamente normalizado */
  rot: Float32Array;
  /** logit da opacidade */
  opacity: Float32Array;
  /** f_dc_0..2 */
  dc: Float32Array;
  /** f_rest_* por gaussiana, canal maior (todos os coeficientes do R, depois G, B). */
  rest: Float32Array;
}

/** Coeficientes SH por canal além do DC, para cada grau. */
export const shCoeffs = (degree: number) => [0, 3, 8, 15][Math.min(degree, 3)];
export const restLen = (s: Splats) => shCoeffs(s.shDegree) * 3;

const TYPES: Record<string, [number, (v: DataView, o: number) => number]> = {
  char: [1, (v, o) => v.getInt8(o)],
  int8: [1, (v, o) => v.getInt8(o)],
  uchar: [1, (v, o) => v.getUint8(o)],
  uint8: [1, (v, o) => v.getUint8(o)],
  short: [2, (v, o) => v.getInt16(o, true)],
  int16: [2, (v, o) => v.getInt16(o, true)],
  ushort: [2, (v, o) => v.getUint16(o, true)],
  uint16: [2, (v, o) => v.getUint16(o, true)],
  int: [4, (v, o) => v.getInt32(o, true)],
  int32: [4, (v, o) => v.getInt32(o, true)],
  uint: [4, (v, o) => v.getUint32(o, true)],
  uint32: [4, (v, o) => v.getUint32(o, true)],
  float: [4, (v, o) => v.getFloat32(o, true)],
  float32: [4, (v, o) => v.getFloat32(o, true)],
  double: [8, (v, o) => v.getFloat64(o, true)],
  float64: [8, (v, o) => v.getFloat64(o, true)],
};

interface Prop {
  name: string;
  size: number;
  read: (v: DataView, o: number) => number;
  offset: number;
}

interface Header {
  count: number;
  props: Prop[];
  stride: number;
  /** Bytes do cabeçalho (início dos dados). */
  dataStart: number;
}

/** Cabeçalho do .ply (o elemento `vertex`; os seguintes são ignorados). */
export function readHeader(bytes: Uint8Array): Header {
  const limit = Math.min(bytes.length, 65536);
  const text = new TextDecoder("latin1").decode(bytes.subarray(0, limit));
  const end = text.indexOf("end_header\n");
  if (!text.startsWith("ply\n") || end < 0) throw new Error("não é um arquivo .ply");
  let count: number | null = null;
  let inVertex = false;
  let stride = 0;
  const props: Prop[] = [];
  for (const line of text.slice(0, end).split("\n").slice(1)) {
    const t = line.trim().split(/\s+/);
    if (t[0] === "format" && t[1] !== "binary_little_endian")
      throw new Error(`formato .ply não suportado: ${t[1]} (só binary_little_endian)`);
    if (t[0] === "element") {
      if (count != null) inVertex = false;
      else if (t[1] === "vertex") {
        count = Number(t[2]);
        inVertex = true;
      } else throw new Error(`.ply com o elemento "${t[1]}" antes dos vértices (compactado?) não é suportado`);
    } else if (t[0] === "property" && inVertex) {
      if (t[1] === "list") throw new Error("propriedade em lista nos vértices do .ply");
      const ty = TYPES[t[1]];
      if (!ty) throw new Error(`tipo de propriedade desconhecido: ${t[1]}`);
      props.push({ name: t[2], size: ty[0], read: ty[1], offset: stride });
      stride += ty[0];
    }
  }
  if (count == null) throw new Error(".ply sem elemento vertex");
  return { count, props, stride, dataStart: end + "end_header\n".length };
}

function degreeFromRest(n: number) {
  const d = [0, 9, 24, 45].indexOf(n);
  if (d < 0) throw new Error(`${n} coeficientes f_rest não correspondem a nenhum grau SH`);
  return d;
}

/** Quantidade de gaussianas e grau SH, lendo só o cabeçalho. */
export function plyInfo(head: Uint8Array) {
  const h = readHeader(head);
  return { count: h.count, shDegree: degreeFromRest(h.props.filter((p) => p.name.startsWith("f_rest_")).length) };
}

export function readPly(bytes: Uint8Array): Splats {
  const { count, props, stride, dataStart } = readHeader(bytes);
  const col = (name: string) => {
    const p = props.find((x) => x.name === name);
    if (!p) throw new Error(`.ply sem a propriedade ${name} (não parece um Gaussian Splat)`);
    return p;
  };
  const cols = (names: string[]) => names.map(col);
  const pos = cols(["x", "y", "z"]);
  const scale = cols(["scale_0", "scale_1", "scale_2"]);
  const rot = cols(["rot_0", "rot_1", "rot_2", "rot_3"]);
  const opacity = col("opacity");
  const dc = cols(["f_dc_0", "f_dc_1", "f_dc_2"]);
  const rest: Prop[] = [];
  for (let p = props.find((x) => x.name === "f_rest_0"); p; p = props.find((x) => x.name === `f_rest_${rest.length}`)) rest.push(p);
  const shDegree = degreeFromRest(rest.length);
  if (dataStart + count * stride > bytes.length) throw new Error(".ply truncado");

  const s: Splats = {
    count,
    shDegree,
    pos: new Float32Array(count * 3),
    scale: new Float32Array(count * 3),
    rot: new Float32Array(count * 4),
    opacity: new Float32Array(count),
    dc: new Float32Array(count * 3),
    rest: new Float32Array(count * rest.length),
  };
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const rl = rest.length;
  for (let i = 0, row = dataStart; i < count; i++, row += stride) {
    for (let a = 0; a < 3; a++) {
      s.pos[i * 3 + a] = pos[a].read(v, row + pos[a].offset);
      s.scale[i * 3 + a] = scale[a].read(v, row + scale[a].offset);
      s.dc[i * 3 + a] = dc[a].read(v, row + dc[a].offset);
    }
    for (let a = 0; a < 4; a++) s.rot[i * 4 + a] = rot[a].read(v, row + rot[a].offset);
    s.opacity[i] = opacity.read(v, row + opacity.offset);
    for (let k = 0; k < rl; k++) s.rest[i * rl + k] = rest[k].read(v, row + rest[k].offset);
  }
  return s;
}

export function writePly(s: Splats): Uint8Array {
  const rl = restLen(s);
  const names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"];
  for (let i = 0; i < rl; i++) names.push(`f_rest_${i}`);
  names.push("opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3");
  const header = new TextEncoder().encode(
    "ply\nformat binary_little_endian 1.0\ncomment Exported from SplatForge\n" +
      `element vertex ${s.count}\n` +
      names.map((n) => `property float ${n}\n`).join("") +
      "end_header\n",
  );
  const stride = names.length;
  const out = new Uint8Array(header.length + s.count * stride * 4);
  out.set(header);
  const v = new DataView(out.buffer, header.length);
  let o = 0;
  const put = (x: number) => {
    v.setFloat32(o, x, true);
    o += 4;
  };
  for (let i = 0; i < s.count; i++) {
    for (let a = 0; a < 3; a++) put(s.pos[i * 3 + a]);
    for (let a = 0; a < 3; a++) put(s.dc[i * 3 + a]);
    for (let k = 0; k < rl; k++) put(s.rest[i * rl + k]);
    put(s.opacity[i]);
    for (let a = 0; a < 3; a++) put(s.scale[i * 3 + a]);
    for (let a = 0; a < 4; a++) put(s.rot[i * 4 + a]);
  }
  return out;
}

/** Mantém só as gaussianas de `keep`, na ordem dada. */
export function select(s: Splats, keep: ArrayLike<number>): Splats {
  const n = keep.length, rl = restLen(s);
  const out: Splats = {
    count: n,
    shDegree: s.shDegree,
    pos: new Float32Array(n * 3),
    scale: new Float32Array(n * 3),
    rot: new Float32Array(n * 4),
    opacity: new Float32Array(n),
    dc: new Float32Array(n * 3),
    rest: new Float32Array(n * rl),
  };
  for (let k = 0; k < n; k++) {
    const i = keep[k];
    out.pos.set(s.pos.subarray(i * 3, i * 3 + 3), k * 3);
    out.scale.set(s.scale.subarray(i * 3, i * 3 + 3), k * 3);
    out.rot.set(s.rot.subarray(i * 4, i * 4 + 4), k * 4);
    out.opacity[k] = s.opacity[i];
    out.dc.set(s.dc.subarray(i * 3, i * 3 + 3), k * 3);
    out.rest.set(s.rest.subarray(i * rl, (i + 1) * rl), k * rl);
  }
  return out;
}

/** Descarta as bandas SH acima de `degree`. */
export function truncateSh(s: Splats, degree: number): Splats {
  if (degree >= s.shDegree) return s;
  const [old, nw] = [shCoeffs(s.shDegree), shCoeffs(degree)];
  const rest = new Float32Array(s.count * nw * 3);
  for (let g = 0; g < s.count; g++)
    for (let c = 0; c < 3; c++)
      rest.set(s.rest.subarray(g * old * 3 + c * old, g * old * 3 + c * old + nw), (g * 3 + c) * nw);
  return { ...s, rest, shDegree: degree };
}

/** Splats determinísticos para os testes (mesmo gerador dos testes em Rust). */
export function sample(n: number, degree: number): Splats {
  const rl = shCoeffs(degree) * 3;
  const M = (1n << 64n) - 1n;
  const f = (i: number, k: number) => {
    let z = ((BigInt(i) * 0x9e3779b97f4a7c15n) & M) ^ ((BigInt(k) * 0xbf58476d1ce4e5b9n) & M);
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & M;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & M;
    return Number(z ^ (z >> 31n)) / 2 ** 64 - 0.5;
  };
  const arr = (len: number, g: (k: number) => number) => Float32Array.from({ length: len }, (_, k) => g(k));
  return {
    count: n,
    shDegree: degree,
    pos: arr(n * 3, (k) => f(Math.floor(k / 3), k) * 4),
    scale: arr(n * 3, (k) => -4 + f(Math.floor(k / 3), k)),
    rot: arr(n * 4, (k) => f(Math.floor(k / 4), k) + (k % 4 === 0 ? 1 : 0)),
    opacity: arr(n, (i) => f(i, 1) * 8),
    dc: arr(n * 3, (k) => f(Math.floor(k / 3), k)),
    rest: arr(n * rl, (k) => f(Math.floor(k / Math.max(rl, 1)), k) * 0.1),
  };
}
