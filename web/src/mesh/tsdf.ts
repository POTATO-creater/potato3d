// Sparse TSDF fusion on WebGPU. Space is split into blocks of 8³ voxels and
// only the blocks near the observed surface are allocated (first pass, on the
// CPU, from the back-projected depth maps); the second pass integrates every
// depth map into those blocks on the GPU. Per voxel: TSDF and weight as two
// f16 (one u32) and the running mean colour as RGBA8 (one u32).

import type { View } from "./colmap.ts";

export const BLOCK = 8;
const BLOCK_VOXELS = BLOCK * BLOCK * BLOCK;
const WG = 64;

/** One depth map at the render resolution, with the photo colour and the
 *  validity (subject mask ∧ splat alpha) packed per pixel. */
export interface DepthFrame {
  view: View;
  width: number;
  height: number;
  /** Intrinsics at this resolution. */
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  /** Camera-space z per pixel (0 = none). */
  depth: Float32Array;
  /** RGBA8 per pixel; alpha 255 = valid, 0 = ignore. */
  rgba: Uint32Array;
}

/** Integer key of a block (coordinates within ±2¹⁵, i.e. ±1.5 km at 6 mm
 *  voxels): exact in a double, unlike wider packings. */
const OFF = 1 << 15;
export const blockKey = (bx: number, by: number, bz: number) => ((bx + OFF) * 65536 + (by + OFF)) * 65536 + (bz + OFF);
const key = blockKey;

export class BlockSet {
  readonly index = new Map<number, number>();
  readonly coords: number[] = [];
  readonly voxel: number;
  constructor(voxel: number) {
    this.voxel = voxel;
  }
  mark(x: number, y: number, z: number) {
    const s = this.voxel * BLOCK;
    const bx = Math.floor(x / s), by = Math.floor(y / s), bz = Math.floor(z / s);
    if (Math.max(Math.abs(bx), Math.abs(by), Math.abs(bz)) >= OFF) return; // far outlier
    const k = key(bx, by, bz);
    if (!this.index.has(k)) {
      this.index.set(k, this.coords.length / 3);
      this.coords.push(bx, by, bz);
    }
  }
  get count() {
    return this.coords.length / 3;
  }
  /** Marks the blocks around the surface seen in `f`: the back-projected point
   *  and the two ends of its truncation band along the ray. `stride` skips
   *  pixels (blocks are much larger than a pixel's footprint). */
  markFrame(f: DepthFrame, trunc: number, stride = 2) {
    const { R, t } = f.view;
    for (let py = 0; py < f.height; py += stride) {
      for (let px = 0; px < f.width; px += stride) {
        const i = py * f.width + px;
        const z = f.depth[i];
        if (!(z > 0) || f.rgba[i] >>> 24 === 0) continue;
        // camera ray (unnormalised, z = 1) and its world-space point at depth z
        const rx = (px + 0.5 - f.cx) / f.fx, ry = (py + 0.5 - f.cy) / f.fy;
        const inv = 1 / Math.hypot(rx, ry, 1);
        for (const s of [-trunc, 0, trunc]) {
          const d = z + s * inv; // move `s` along the unit ray
          const cx = rx * d - t[0], cy = ry * d - t[1], cz = d - t[2];
          // world = Rᵀ (cam − t)
          this.mark(
            R[0] * cx + R[3] * cy + R[6] * cz,
            R[1] * cx + R[4] * cy + R[7] * cz,
            R[2] * cx + R[5] * cy + R[8] * cz,
          );
        }
      }
    }
  }
}

const SHADER = /* wgsl */ `
struct Params {
  rot0: vec4f, rot1: vec4f, rot2: vec4f, trans: vec4f,
  // fx, fy, cx, cy
  intr: vec4f,
  // voxel, trunc, width, height
  grid: vec4f,
  nblocks: u32, row: u32, pad0: u32, pad1: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> blocks: array<i32>;
@group(0) @binding(2) var<storage, read> depth: array<f32>;
@group(0) @binding(3) var<storage, read> rgba: array<u32>;
@group(0) @binding(4) var<storage, read_write> tsdfw: array<u32>;
@group(0) @binding(5) var<storage, read_write> color: array<u32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * P.row;
  if (i >= P.nblocks * ${BLOCK_VOXELS}u) { return; }
  let b = i / ${BLOCK_VOXELS}u;
  let l = i % ${BLOCK_VOXELS}u;
  let g = vec3f(f32(blocks[3u * b]), f32(blocks[3u * b + 1u]), f32(blocks[3u * b + 2u])) * ${BLOCK}.0
        + vec3f(f32(l % ${BLOCK}u), f32((l / ${BLOCK}u) % ${BLOCK}u), f32(l / ${BLOCK * BLOCK}u));
  let p = g * P.grid.x;                         // lattice point (world)
  let c = vec3f(dot(P.rot0.xyz, p), dot(P.rot1.xyz, p), dot(P.rot2.xyz, p)) + P.trans.xyz;
  if (c.z <= 1e-4) { return; }
  let u = P.intr.x * c.x / c.z + P.intr.z;
  let v = P.intr.y * c.y / c.z + P.intr.w;
  let w = P.grid.z; let h = P.grid.w;
  if (u < 0.0 || v < 0.0 || u >= w || v >= h) { return; }
  let px = u32(u) + u32(v) * u32(w);
  let col = rgba[px];
  let d = depth[px];
  if ((col >> 24u) == 0u || d <= 0.0) { return; }
  let sdf = d - c.z;
  let trunc = P.grid.y;
  if (sdf < -trunc) { return; }
  let ts = min(1.0, sdf / trunc);
  let old = unpack2x16float(tsdfw[i]);
  let nw = old.y + 1.0;
  tsdfw[i] = pack2x16float(vec2f((old.x * old.y + ts) / nw, min(nw, 60000.0)));
  let oc = unpack4x8unorm(color[i]);
  let nc = unpack4x8unorm(col);
  color[i] = pack4x8unorm(vec4f((oc.rgb * old.y + nc.rgb) / nw, 1.0));
}
`;

export interface Volume {
  voxel: number;
  trunc: number;
  /** Block coordinates (bx, by, bz) per allocated block. */
  blocks: Int32Array;
  /** Per voxel (block-major, x fastest): pack2x16float(tsdf, weight). */
  tsdfw: Uint32Array;
  /** Per voxel: RGBA8 mean colour (sRGB, as in the photos). */
  color: Uint32Array;
}

/** Integrates depth frames into the blocks of `set`. Call `integrate` per
 *  frame, then `read` once. */
export class TsdfGpu {
  private pipeline: GPUComputePipeline;
  private blocks: GPUBuffer;
  private tsdfw: GPUBuffer;
  private color: GPUBuffer;
  private params: GPUBuffer;
  private nblocks: number;
  private depthBuf: GPUBuffer | null = null;
  private rgbaBuf: GPUBuffer | null = null;
  private device: GPUDevice;
  private set: BlockSet;
  readonly trunc: number;

  constructor(device: GPUDevice, set: BlockSet, trunc: number) {
    this.device = device;
    this.set = set;
    this.trunc = trunc;
    this.nblocks = set.count;
    const voxels = this.nblocks * BLOCK_VOXELS;
    const limit = device.limits.maxStorageBufferBindingSize;
    if (voxels * 4 > limit) throw new Error(`mesh volume too large for this GPU (${(voxels * 4 / 2 ** 20) | 0} MB per buffer)`);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.blocks = device.createBuffer({ size: Math.max(16, this.nblocks * 12), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.blocks, 0, new Int32Array(set.coords));
    this.tsdfw = device.createBuffer({ size: voxels * 4, usage: storage }); // zero = weight 0
    this.color = device.createBuffer({ size: voxels * 4, usage: storage });
    this.params = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: "main" },
    });
  }

  integrate(f: DepthFrame) {
    const d = this.device;
    const px = f.width * f.height;
    if (!this.depthBuf || this.depthBuf.size < px * 4) {
      this.depthBuf?.destroy();
      this.rgbaBuf?.destroy();
      this.depthBuf = d.createBuffer({ size: px * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.rgbaBuf = d.createBuffer({ size: px * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    d.queue.writeBuffer(this.depthBuf, 0, f.depth as BufferSource);
    d.queue.writeBuffer(this.rgbaBuf!, 0, f.rgba as BufferSource);
    const total = this.nblocks * BLOCK_VOXELS;
    const groups = Math.ceil(total / WG);
    const gx = Math.min(groups, 65535), gy = Math.ceil(groups / gx);
    const { R, t } = f.view;
    const p = new ArrayBuffer(112);
    const fv = new Float32Array(p, 0, 24);
    fv.set([R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0, t[0], t[1], t[2], 0]);
    fv.set([f.fx, f.fy, f.cx, f.cy, this.set.voxel, this.trunc, f.width, f.height], 16);
    new Uint32Array(p, 96, 4).set([this.nblocks, gx * WG, 0, 0]);
    d.queue.writeBuffer(this.params, 0, p);
    const bind = d.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [this.params, this.blocks, this.depthBuf, this.rgbaBuf!, this.tsdfw, this.color].map((buffer, binding) => ({
        binding,
        resource: { buffer: buffer! },
      })),
    });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
    d.queue.submit([enc.finish()]);
  }

  async read(): Promise<Volume> {
    const d = this.device;
    const size = this.tsdfw.size;
    const read = async (src: GPUBuffer) => {
      const dst = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, dst, 0, size);
      d.queue.submit([enc.finish()]);
      await dst.mapAsync(GPUMapMode.READ);
      const out = new Uint32Array(dst.getMappedRange().slice(0));
      dst.unmap();
      dst.destroy();
      return out;
    };
    const tsdfw = await read(this.tsdfw);
    const color = await read(this.color);
    return { voxel: this.set.voxel, trunc: this.trunc, blocks: new Int32Array(this.set.coords), tsdfw, color };
  }

  destroy() {
    for (const b of [this.blocks, this.tsdfw, this.color, this.params, this.depthBuf, this.rgbaBuf]) b?.destroy();
  }
}
