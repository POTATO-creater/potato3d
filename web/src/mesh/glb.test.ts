// pnpm test:web
import assert from "node:assert/strict";
import { test } from "node:test";
import { writeGlb } from "./glb.ts";

test("GLB: header, JSON and binary chunks agree with the accessors", () => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const colors = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const indices = new Uint32Array([0, 1, 2]);
  const glb = writeGlb({ positions, normals, colors, indices, texture: { bytes: new Uint8Array([1, 2, 3]), mimeType: "image/png" } });
  const dv = new DataView(glb.buffer);
  assert.equal(dv.getUint32(0, true), 0x46546c67);
  assert.equal(dv.getUint32(8, true), glb.length);
  const jsonLen = dv.getUint32(12, true);
  assert.equal(jsonLen % 4, 0);
  const json = JSON.parse(new TextDecoder().decode(glb.subarray(20, 20 + jsonLen)));
  const binLen = dv.getUint32(20 + jsonLen, true);
  assert.equal(dv.getUint32(24 + jsonLen, true), 0x004e4942);
  assert.equal(binLen, json.buffers[0].byteLength);
  const bin = glb.subarray(28 + jsonLen);
  const prim = json.meshes[0].primitives[0];
  const read = (acc: number) => {
    const a = json.accessors[acc], v = json.bufferViews[a.bufferView];
    const bytes = bin.slice(v.byteOffset, v.byteOffset + v.byteLength);
    return a.componentType === 5125 ? new Uint32Array(bytes.buffer) : new Float32Array(bytes.buffer);
  };
  assert.deepEqual(Array.from(read(prim.attributes.POSITION)), Array.from(positions));
  assert.deepEqual(Array.from(read(prim.attributes.COLOR_0)), Array.from(colors));
  assert.deepEqual(Array.from(read(prim.indices)), [0, 1, 2]);
  assert.deepEqual(json.accessors[prim.attributes.POSITION].max, [1, 1, 0]);
  assert.equal(json.images[0].mimeType, "image/png");
  assert.equal(json.materials[0].pbrMetallicRoughness.baseColorTexture.index, 0);
  for (const v of json.bufferViews) assert.equal(v.byteOffset % 4, 0);
});
