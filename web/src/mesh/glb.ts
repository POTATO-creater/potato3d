// Minimal binary glTF 2.0 writer: one mesh, one primitive; and the mesh
// export's placement (y-up, centred, on the floor).

import type { BuiltMesh } from "./process.ts";

export interface GlbMesh {
  positions: Float32Array;
  normals?: Float32Array;
  /** Linear RGB per vertex (glTF COLOR_0 is linear). */
  colors?: Float32Array;
  uvs?: Float32Array;
  indices: Uint32Array;
  /** Base colour texture (JPEG or PNG bytes). */
  texture?: { bytes: Uint8Array; mimeType: "image/jpeg" | "image/png" };
}

const pad4 = (n: number) => (n + 3) & ~3;

export function writeGlb(m: GlbMesh): Uint8Array {
  const views: { buffer: 0; byteOffset: number; byteLength: number; target?: number }[] = [];
  const accessors: Record<string, unknown>[] = [];
  const parts: Uint8Array[] = [];
  let offset = 0;
  const addView = (bytes: Uint8Array, target?: number) => {
    views.push({ buffer: 0, byteOffset: offset, byteLength: bytes.byteLength, ...(target ? { target } : {}) });
    parts.push(bytes);
    const padded = pad4(bytes.byteLength);
    if (padded > bytes.byteLength) parts.push(new Uint8Array(padded - bytes.byteLength));
    offset += padded;
    return views.length - 1;
  };
  const vec = (data: Float32Array, size: number, type: string, minmax = false) => {
    const view = addView(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 34962);
    const acc: Record<string, unknown> = { bufferView: view, componentType: 5126, count: data.length / size, type };
    if (minmax) {
      const min = Array(size).fill(Infinity), max = Array(size).fill(-Infinity);
      for (let i = 0; i < data.length; i++) {
        const k = i % size;
        if (data[i] < min[k]) min[k] = data[i];
        if (data[i] > max[k]) max[k] = data[i];
      }
      Object.assign(acc, { min, max });
    }
    accessors.push(acc);
    return accessors.length - 1;
  };

  const attributes: Record<string, number> = { POSITION: vec(m.positions, 3, "VEC3", true) };
  if (m.normals) attributes.NORMAL = vec(m.normals, 3, "VEC3");
  if (m.colors) attributes.COLOR_0 = vec(m.colors, 3, "VEC3");
  if (m.uvs) attributes.TEXCOORD_0 = vec(m.uvs, 2, "VEC2");
  const iv = addView(new Uint8Array(m.indices.buffer, m.indices.byteOffset, m.indices.byteLength), 34963);
  accessors.push({ bufferView: iv, componentType: 5125, count: m.indices.length, type: "SCALAR" });
  const indices = accessors.length - 1;

  const json: Record<string, unknown> = {
    asset: { version: "2.0", generator: "Gausify Orbit" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes, indices, material: 0 }] }],
    materials: [
      {
        pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 0.8 },
        doubleSided: false,
      },
    ],
    accessors,
    bufferViews: views,
  };
  if (m.texture) {
    const view = addView(m.texture.bytes);
    json.images = [{ bufferView: view, mimeType: m.texture.mimeType }];
    json.samplers = [{ magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 }];
    json.textures = [{ source: 0, sampler: 0 }];
    (json.materials as Record<string, Record<string, unknown>>[])[0].pbrMetallicRoughness.baseColorTexture = { index: 0 };
  }
  json.buffers = [{ byteLength: offset }];

  const text = new TextEncoder().encode(JSON.stringify(json));
  const jsonLen = pad4(text.length);
  const total = 12 + 8 + jsonLen + 8 + offset;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546c67, true); // "glTF"
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonLen, true);
  dv.setUint32(16, 0x4e4f534a, true); // "JSON"
  out.set(text, 20);
  out.fill(0x20, 20 + text.length, 20 + jsonLen); // JSON pads with spaces
  let o = 20 + jsonLen;
  dv.setUint32(o, offset, true);
  dv.setUint32(o + 4, 0x004e4942, true); // "BIN\0"
  o += 8;
  for (const p of parts) {
    out.set(p, o);
    o += p.byteLength;
  }
  return out;
}

/** sRGB (0–1) → linear, for glTF vertex colours. */
export function srgbToLinear(c: number) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
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

/** The GLB: rotated so `up` is +Y, centred on the vertical axis and standing
 *  on y = 0; vertex colours, or the baked texture (JPEG) instead. */
export function writeMeshGlb(m: BuiltMesh, up: [number, number, number], texture?: Uint8Array): Uint8Array {
  const R = yUp(up);
  const P = m.positions;
  const out = new Float32Array(P.length), normals = new Float32Array(P.length);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < P.length; v += 3) {
    for (let r = 0; r < 3; r++) {
      const x = R[r][0] * P[v] + R[r][1] * P[v + 1] + R[r][2] * P[v + 2];
      out[v + r] = x;
      normals[v + r] = R[r][0] * m.normals[v] + R[r][1] * m.normals[v + 1] + R[r][2] * m.normals[v + 2];
      lo[r] = Math.min(lo[r], x);
      hi[r] = Math.max(hi[r], x);
    }
  }
  const shift = [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2];
  for (let v = 0; v < out.length; v += 3) for (let r = 0; r < 3; r++) out[v + r] -= shift[r];
  // glTF multiplies COLOR_0 into the texture: one or the other
  return writeGlb(
    texture && m.uvs
      ? { positions: out, normals, uvs: m.uvs, indices: m.indices, texture: { bytes: texture, mimeType: "image/jpeg" } }
      : { positions: out, normals, colors: m.colors.map(srgbToLinear), indices: m.indices },
  );
}
