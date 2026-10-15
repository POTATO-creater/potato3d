// Minimal binary glTF 2.0 writer: one mesh, one primitive.

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
