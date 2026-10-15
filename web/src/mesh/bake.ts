// Photo texture bake on WebGPU. Every texel of the UV atlas is a point on the
// mesh; its colour is a weighted mean of the photos that see it:
// - visible: the mesh rendered from the photo's camera, then a min-pool of
//   that depth (6 px): nothing nearer may lie close by, so colour from an arm
//   in front never bleeds onto the body behind it;
// - weight cos⁶ of the viewing angle (frontal views dominate), times a
//   feather that fades each photo out over the last 40 px of its subject
//   mask (seams between photos become gradients instead of steps);
// - robust: a second pass drops views more than 0.12 away from the first
//   pass's mean when at least 3 views saw the texel (reflections, a hand
//   passing in front);
// - texels only seen at grazing angles keep the TSDF's vertex colour;
//   chart borders are dilated so filtering never reads the empty background.

import type { View } from "./colmap.ts";
import type { UvMesh } from "./uv.ts";

/** Min-pool radius (px) of the occlusion test. */
const OCCLUDER_PX = 6;
/** Feather width (px at full resolution; computed at ¼). */
const FEATHER_PX = 40;
const Q = 4;
/** A view more than this from the first pass's mean colour is dropped. */
const OUTLIER = 0.12;
/** Below cos⁶ = 0.4⁶ a texel counts as unseen and keeps its vertex colour. */
const RAW_MIN = 0.4 ** 6, RAW_FULL = 0.6 ** 6;
const GUTTER = 8;

const COMMON = /* wgsl */ `
struct Cam {
  rot0: vec4f, rot1: vec4f, rot2: vec4f, trans: vec4f,
  intr: vec4f,      // fx, fy, cx, cy
  dims: vec4f,      // W, H, W/Q, H/Q
  params: vec4f,    // occlusion tolerance, near, far, has mask
  center: vec4f,    // camera centre (world)
};
@group(0) @binding(0) var<uniform> cam: Cam;
fn toCam(p: vec3f) -> vec3f {
  return vec3f(dot(cam.rot0.xyz, p), dot(cam.rot1.xyz, p), dot(cam.rot2.xyz, p)) + cam.trans.xyz;
}
struct VIn { @location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f, @location(3) color: vec3f };
`;

const DEPTH = /* wgsl */ `${COMMON}
struct VOut { @builtin(position) clip: vec4f, @location(0) z: f32 };
@vertex fn vs(v: VIn) -> VOut {
  let c = toCam(v.pos);
  let near = cam.params.y; let far = cam.params.z;
  let px = cam.intr.x * c.x + cam.intr.z * c.z;   // pixel × z
  let py = cam.intr.y * c.y + cam.intr.w * c.z;
  var o: VOut;
  o.clip = vec4f(2.0 * px / cam.dims.x - c.z, c.z - 2.0 * py / cam.dims.y, far / (far - near) * (c.z - near), c.z);
  o.z = c.z;
  return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4f { return vec4f(i.z, 0.0, 0.0, 0.0); }
`;

const FILTER = /* wgsl */ `${COMMON}
@group(0) @binding(1) var depthTex: texture_2d<f32>;
@group(0) @binding(2) var maskTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(4) var<storage, read_write> minDepth: array<f32>;
@group(0) @binding(5) var<storage, read_write> feather: array<f32>;
const R = ${OCCLUDER_PX}i;
const F = ${FEATHER_PX / Q}i;
fn depthAt(x: i32, y: i32) -> f32 {
  let z = textureLoad(depthTex, vec2i(x, y), 0).r;
  return select(1e30, z, z > 0.0);
}
@compute @workgroup_size(16, 16) fn minH(@builtin(global_invocation_id) g: vec3u) {
  let w = i32(cam.dims.x); let h = i32(cam.dims.y);
  let x = i32(g.x); let y = i32(g.y);
  if (x >= w || y >= h) { return; }
  var m = 1e30;
  for (var d = -R; d <= R; d++) { m = min(m, depthAt(clamp(x + d, 0, w - 1), y)); }
  tmp[y * w + x] = m;
}
@compute @workgroup_size(16, 16) fn minV(@builtin(global_invocation_id) g: vec3u) {
  let w = i32(cam.dims.x); let h = i32(cam.dims.y);
  let x = i32(g.x); let y = i32(g.y);
  if (x >= w || y >= h) { return; }
  var m = 1e30;
  for (var d = -R; d <= R; d++) { m = min(m, tmp[clamp(y + d, 0, h - 1) * w + x]); }
  minDepth[y * w + x] = m;
}
// subject (mask on, mesh hit) at ¼ resolution
fn inside(x: i32, y: i32) -> bool {
  let w = i32(cam.dims.z); let h = i32(cam.dims.w);
  if (x < 0 || y < 0 || x >= w || y >= h) { return false; }
  let p = vec2i(x * ${Q} + ${Q / 2}, y * ${Q} + ${Q / 2});
  let hit = textureLoad(depthTex, p, 0).r > 0.0;
  let m = select(1.0, textureLoad(maskTex, p, 0).r, cam.params.w > 0.5);
  return hit && m > 0.5;
}
// exact distance to the outside, capped at F: horizontal then vertical pass
@compute @workgroup_size(16, 16) fn edtH(@builtin(global_invocation_id) g: vec3u) {
  let w = i32(cam.dims.z); let h = i32(cam.dims.w);
  let x = i32(g.x); let y = i32(g.y);
  if (x >= w || y >= h) { return; }
  var best = f32(F + 1);
  for (var d = -F; d <= F; d++) { if (!inside(x + d, y)) { best = min(best, abs(f32(d))); } }
  tmp[y * w + x] = best;
}
@compute @workgroup_size(16, 16) fn edtV(@builtin(global_invocation_id) g: vec3u) {
  let w = i32(cam.dims.z); let h = i32(cam.dims.w);
  let x = i32(g.x); let y = i32(g.y);
  if (x >= w || y >= h) { return; }
  var best = 1e9;
  for (var d = -F; d <= F; d++) {
    let yy = y + d;
    let gx = select(0.0, tmp[clamp(yy, 0, h - 1) * w + x], yy >= 0 && yy < h);
    best = min(best, gx * gx + f32(d * d));
  }
  feather[y * w + x] = min(sqrt(best), f32(F)) / f32(F);
}
`;

const TEXEL = /* wgsl */ `${COMMON}
@group(0) @binding(1) var photo: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> minDepth: array<f32>;
@group(0) @binding(4) var<storage, read> feather: array<f32>;
@group(0) @binding(5) var meanTex: texture_2d<f32>;
@group(0) @binding(6) var countTex: texture_2d<f32>;
struct VOut { @builtin(position) clip: vec4f, @location(0) p: vec3f, @location(1) n: vec3f, @location(2) color: vec3f };
@vertex fn vs(v: VIn) -> VOut {
  var o: VOut;
  o.clip = vec4f(v.uv.x * 2.0 - 1.0, 1.0 - v.uv.y * 2.0, 0.5, 1.0);
  o.p = v.pos; o.n = v.normal; o.color = v.color;
  return o;
}
struct Seen { ok: bool, rgb: vec3f, raw: f32, w: f32 };
fn look(p: vec3f, n: vec3f) -> Seen {
  var s: Seen; s.ok = false;
  let c = toCam(p);
  if (c.z <= 0.0) { return s; }
  let u = cam.intr.x * c.x / c.z + cam.intr.z;
  let v = cam.intr.y * c.y / c.z + cam.intr.w;
  if (u < 0.0 || v < 0.0 || u >= cam.dims.x || v >= cam.dims.y) { return s; }
  let ix = u32(u); let iy = u32(v);
  let zmin = minDepth[iy * u32(cam.dims.x) + ix];
  if (zmin > 1e29 || c.z - zmin > cam.params.x) { return s; }
  let f = feather[min(iy / ${Q}u, u32(cam.dims.w) - 1u) * u32(cam.dims.z) + min(ix / ${Q}u, u32(cam.dims.z) - 1u)];
  if (f <= 0.0) { return s; }
  let cosv = dot(normalize(n), normalize(cam.center.xyz - p));
  if (cosv <= 0.0) { return s; }
  let c2 = cosv * cosv;
  s.raw = c2 * c2 * c2;
  s.w = s.raw * max(smoothstep(0.0, 1.0, f), 1e-3);
  s.rgb = textureSampleLevel(photo, samp, vec2f(u / cam.dims.x, v / cam.dims.y), 0.0).rgb;
  s.ok = true;
  return s;
}
struct Out { @location(0) acc: vec4f, @location(1) extra: vec4f };
@fragment fn first(i: VOut) -> Out {
  let s = look(i.p, i.n);
  if (!s.ok) { discard; }
  return Out(vec4f(s.rgb * s.w, s.w), vec4f(1.0, 0.0, 0.0, 0.0));
}
@fragment fn second(i: VOut) -> Out {
  let s = look(i.p, i.n);
  if (!s.ok) { discard; }
  let t = vec2i(i.clip.xy);
  let m = textureLoad(meanTex, t, 0);
  let d = abs(s.rgb - m.rgb);
  if (textureLoad(countTex, t, 0).r >= 3.0 && max(d.r, max(d.g, d.b)) > ${OUTLIER}) { discard; }
  return Out(vec4f(s.rgb * s.w, s.w), vec4f(s.raw, 0.0, 0.0, 0.0));
}
@fragment fn base(i: VOut) -> @location(0) vec4f { return vec4f(i.color, 1.0); }
`;

const RESOLVE = /* wgsl */ `
@group(0) @binding(0) var acc: texture_2d<f32>;
@group(0) @binding(1) var extra: texture_2d<f32>;
@group(0) @binding(2) var base: texture_2d<f32>;
@group(0) @binding(3) var src: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var meanOut: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(16, 16) fn mean(@builtin(global_invocation_id) g: vec3u) {
  let t = vec2i(g.xy);
  if (any(g.xy >= textureDimensions(acc))) { return; }
  let a = textureLoad(acc, t, 0);
  textureStore(meanOut, t, vec4f(a.rgb / max(a.a, 1e-8), 1.0));
}
// alpha: 1 photo colour, 0.8 vertex colour only, 0 empty
@compute @workgroup_size(16, 16) fn compose(@builtin(global_invocation_id) g: vec3u) {
  let t = vec2i(g.xy);
  if (any(g.xy >= textureDimensions(acc))) { return; }
  let a = textureLoad(acc, t, 0);
  let raw = textureLoad(extra, t, 0).r;
  let b = textureLoad(base, t, 0);
  let k = select(0.0, smoothstep(${RAW_MIN}, ${RAW_FULL}, raw), a.a > 0.0);
  let rgb = mix(b.rgb, a.rgb / max(a.a, 1e-8), k);
  let alpha = select(select(0.0, 0.8, b.a > 0.0), 1.0, k > 0.0 && b.a > 0.0);
  textureStore(dst, t, vec4f(rgb, alpha));
}
// gutter: empty texels take the mean of their filled neighbours
@compute @workgroup_size(16, 16) fn dilate(@builtin(global_invocation_id) g: vec3u) {
  let t = vec2i(g.xy);
  let size = vec2i(textureDimensions(src));
  if (any(t >= size)) { return; }
  let c = textureLoad(src, t, 0);
  if (c.a > 0.0) { textureStore(dst, t, c); return; }
  var sum = vec3f(0.0); var n = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = clamp(t + vec2i(dx, dy), vec2i(0), size - 1);
      let s = textureLoad(src, q, 0);
      if (s.a > 0.0) { sum += s.rgb; n += 1.0; }
    }
  }
  textureStore(dst, t, select(vec4f(0.0), vec4f(sum / max(n, 1.0), 0.5), n > 0.0));
}
`;

export interface BakeStats {
  size: number;
  /** Share of the mesh's texels coloured from photos (the rest: vertex colour). */
  photoCoverage: number;
}

const VERTEX_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: 44,
  attributes: [
    { shaderLocation: 0, offset: 0, format: "float32x3" },
    { shaderLocation: 1, offset: 12, format: "float32x3" },
    { shaderLocation: 2, offset: 24, format: "float32x2" },
    { shaderLocation: 3, offset: 32, format: "float32x3" },
  ],
};

/** Bakes the photos into the atlas of `mesh`: `photo(view, pass)` for every
 *  view in pass 0 (consensus) and pass 1 (robust), then `finish`. */
export class TextureBaker {
  private d: GPUDevice;
  private size: number;
  private tolerance: number;
  private vertices: GPUBuffer;
  private indices: GPUBuffer;
  private count: number;
  private camBuf: GPUBuffer;
  private depthPipe: GPURenderPipeline;
  private filter: Record<"minH" | "minV" | "edtH" | "edtV", GPUComputePipeline>;
  private filterLayout: GPUBindGroupLayout;
  private texelLayout: GPUBindGroupLayout;
  private texelPipes: GPURenderPipeline[];
  private basePipe: GPURenderPipeline;
  private resolveLayout: GPUBindGroupLayout;
  private resolvePipes: Record<"mean" | "compose" | "dilate", GPUComputePipeline>;
  private sampler: GPUSampler;
  private acc: GPUTexture;
  private extra: GPUTexture;
  private mean: GPUTexture;
  private count0: GPUTexture;
  private baseTex: GPUTexture;
  private bounds: { center: number[]; radius: number };
  private frame: {
    w: number;
    h: number;
    depth: GPUTexture;
    zbuf: GPUTexture;
    tmp: GPUBuffer;
    minDepth: GPUBuffer;
    feather: GPUBuffer;
  } | null = null;
  private pass = 0;
  private frameTextures: GPUTexture[] = [];

  constructor(device: GPUDevice, mesh: UvMesh, size: number, tolerance: number) {
    const d = (this.d = device);
    this.size = size;
    this.tolerance = tolerance;
    const n = mesh.positions.length / 3;
    const vtx = new Float32Array(11 * n);
    for (let i = 0; i < n; i++) {
      vtx.set(mesh.positions.subarray(3 * i, 3 * i + 3), 11 * i);
      vtx.set(mesh.normals.subarray(3 * i, 3 * i + 3), 11 * i + 3);
      vtx.set(mesh.uvs.subarray(2 * i, 2 * i + 2), 11 * i + 6);
      vtx.set(mesh.colors.subarray(3 * i, 3 * i + 3), 11 * i + 8);
    }
    this.vertices = d.createBuffer({ size: vtx.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.vertices, 0, vtx);
    this.indices = d.createBuffer({ size: mesh.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.indices, 0, mesh.indices as BufferSource);
    this.count = mesh.indices.length;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < mesh.positions.length; i++) {
      lo[i % 3] = Math.min(lo[i % 3], mesh.positions[i]);
      hi[i % 3] = Math.max(hi[i % 3], mesh.positions[i]);
    }
    this.bounds = { center: lo.map((x, k) => (x + hi[k]) / 2), radius: Math.hypot(...hi.map((x, k) => x - lo[k])) / 2 };
    this.camBuf = d.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = d.createSampler({ magFilter: "linear", minFilter: "linear" });

    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const depthModule = d.createShaderModule({ code: DEPTH });
    const camOnly = d.createBindGroupLayout({ entries: [{ binding: 0, visibility: VF, buffer: {} }] });
    this.depthPipe = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [camOnly] }),
      vertex: { module: depthModule, entryPoint: "vs", buffers: [VERTEX_LAYOUT] },
      fragment: { module: depthModule, entryPoint: "fs", targets: [{ format: "r32float" }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
    });

    const C = GPUShaderStage.COMPUTE;
    this.filterLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: {} },
        { binding: 1, visibility: C, texture: { sampleType: "unfilterable-float" } },
        { binding: 2, visibility: C, texture: { sampleType: "float" } },
        ...[3, 4, 5].map((binding) => ({ binding, visibility: C, buffer: { type: "storage" as const } })),
      ],
    });
    const filterModule = d.createShaderModule({ code: FILTER });
    const fl = d.createPipelineLayout({ bindGroupLayouts: [this.filterLayout] });
    this.filter = Object.fromEntries(
      (["minH", "minV", "edtH", "edtV"] as const).map((e) => [e, d.createComputePipeline({ layout: fl, compute: { module: filterModule, entryPoint: e } })]),
    ) as TextureBaker["filter"];

    this.texelLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: VF, buffer: {} },
        { binding: 1, visibility: VF, texture: { sampleType: "float" } },
        { binding: 2, visibility: VF, sampler: {} },
        { binding: 3, visibility: VF, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: VF, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: VF, texture: { sampleType: "unfilterable-float" } },
        { binding: 6, visibility: VF, texture: { sampleType: "unfilterable-float" } },
      ],
    });
    const texelModule = d.createShaderModule({ code: TEXEL });
    const tl = d.createPipelineLayout({ bindGroupLayouts: [this.texelLayout] });
    const add: GPUBlendState = { color: { operation: "add", srcFactor: "one", dstFactor: "one" }, alpha: { operation: "add", srcFactor: "one", dstFactor: "one" } };
    const max: GPUBlendState = { color: { operation: "max", srcFactor: "one", dstFactor: "one" }, alpha: { operation: "max", srcFactor: "one", dstFactor: "one" } };
    this.texelPipes = (["first", "second"] as const).map((entryPoint, k) =>
      d.createRenderPipeline({
        layout: tl,
        vertex: { module: texelModule, entryPoint: "vs", buffers: [VERTEX_LAYOUT] },
        fragment: {
          module: texelModule,
          entryPoint,
          targets: [
            { format: "rgba16float", blend: add },
            { format: "r16float", blend: k ? max : add },
          ],
        },
        primitive: { topology: "triangle-list", cullMode: "none" },
      }),
    );
    this.basePipe = d.createRenderPipeline({
      layout: tl,
      vertex: { module: texelModule, entryPoint: "vs", buffers: [VERTEX_LAYOUT] },
      fragment: { module: texelModule, entryPoint: "base", targets: [{ format: "rgba8unorm" }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
    });

    this.resolveLayout = d.createBindGroupLayout({
      entries: [
        ...[0, 1, 2, 3].map((binding) => ({ binding, visibility: C, texture: { sampleType: "unfilterable-float" as const } })),
        { binding: 4, visibility: C, storageTexture: { access: "write-only", format: "rgba8unorm" } },
        { binding: 5, visibility: C, storageTexture: { access: "write-only", format: "rgba16float" } },
      ],
    });
    const resolveModule = d.createShaderModule({ code: RESOLVE });
    const rl = d.createPipelineLayout({ bindGroupLayouts: [this.resolveLayout] });
    this.resolvePipes = Object.fromEntries(
      (["mean", "compose", "dilate"] as const).map((e) => [e, d.createComputePipeline({ layout: rl, compute: { module: resolveModule, entryPoint: e } })]),
    ) as TextureBaker["resolvePipes"];

    const T = GPUTextureUsage;
    const tex = (format: GPUTextureFormat, usage: number) => d.createTexture({ size: [size, size], format, usage });
    this.acc = tex("rgba16float", T.RENDER_ATTACHMENT | T.TEXTURE_BINDING);
    this.extra = tex("r16float", T.RENDER_ATTACHMENT | T.TEXTURE_BINDING);
    this.count0 = tex("r16float", T.RENDER_ATTACHMENT | T.TEXTURE_BINDING);
    this.mean = tex("rgba16float", T.STORAGE_BINDING | T.TEXTURE_BINDING);
    this.baseTex = tex("rgba8unorm", T.RENDER_ATTACHMENT | T.TEXTURE_BINDING);

    // vertex colours under everything, and empty pass-0 targets
    const enc = d.createCommandEncoder();
    const bg = this.texelBindGroup(null, null);
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.baseTex.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] });
    rp.setPipeline(this.basePipe);
    rp.setBindGroup(0, bg);
    this.draw(rp);
    rp.end();
    this.clearAccumulators(enc, this.count0);
    d.queue.submit([enc.finish()]);
  }

  private draw(rp: GPURenderPassEncoder) {
    rp.setVertexBuffer(0, this.vertices);
    rp.setIndexBuffer(this.indices, "uint32");
    rp.drawIndexed(this.count);
  }

  private clearAccumulators(enc: GPUCommandEncoder, second: GPUTexture) {
    enc
      .beginRenderPass({
        colorAttachments: [this.acc, second].map((t) => ({ view: t.createView(), loadOp: "clear" as const, storeOp: "store" as const, clearValue: [0, 0, 0, 0] })),
      })
      .end();
  }

  /** Bind group of the texel passes; without a frame, placeholders. */
  private texelBindGroup(photo: GPUTexture | null, f: TextureBaker["frame"]) {
    const d = this.d;
    const dummyTex = photo ?? d.createTexture({ size: [1, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING });
    const dummyBuf = (b?: GPUBuffer) => b ?? d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    return d.createBindGroup({
      layout: this.texelLayout,
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: dummyTex.createView() },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: { buffer: dummyBuf(f?.minDepth) } },
        { binding: 4, resource: { buffer: dummyBuf(f?.feather) } },
        // the first pass draws into count0: it can't be read at the same time
        { binding: 5, resource: this.mean.createView() },
        { binding: 6, resource: (f && !this.pass ? this.placeholder() : this.count0).createView() },
      ],
    });
  }

  private placeholderTex: GPUTexture | null = null;
  private placeholder() {
    this.placeholderTex ??= this.d.createTexture({ size: [1, 1], format: "r16float", usage: GPUTextureUsage.TEXTURE_BINDING });
    return this.placeholderTex;
  }

  private frameFor(w: number, h: number) {
    if (this.frame && this.frame.w === w && this.frame.h === h) return this.frame;
    this.releaseFrame();
    const d = this.d, T = GPUTextureUsage;
    const qw = Math.ceil(w / Q), qh = Math.ceil(h / Q);
    const buf = (n: number) => d.createBuffer({ size: 4 * n, usage: GPUBufferUsage.STORAGE });
    this.frame = {
      w,
      h,
      depth: d.createTexture({ size: [w, h], format: "r32float", usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING }),
      zbuf: d.createTexture({ size: [w, h], format: "depth32float", usage: T.RENDER_ATTACHMENT }),
      tmp: buf(w * h),
      minDepth: buf(w * h),
      feather: buf(qw * qh),
    };
    return this.frame;
  }

  private releaseFrame() {
    if (!this.frame) return;
    for (const r of [this.frame.depth, this.frame.zbuf, this.frame.tmp, this.frame.minDepth, this.frame.feather]) r.destroy();
    this.frame = null;
  }

  /** Switches from the consensus pass to the robust one. */
  startSecondPass() {
    const enc = this.d.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(this.resolvePipes.mean);
    cp.setBindGroup(0, this.resolveBindGroup(this.acc, this.baseTex));
    cp.dispatchWorkgroups(Math.ceil(this.size / 16), Math.ceil(this.size / 16));
    cp.end();
    this.clearAccumulators(enc, this.extra);
    this.d.queue.submit([enc.finish()]);
    this.pass = 1;
  }

  private resolveBindGroup(src: GPUTexture, dst: GPUTexture | null) {
    const d = this.d;
    const out = dst && dst.format === "rgba8unorm" && dst.usage & GPUTextureUsage.STORAGE_BINDING ? dst : this.scratch(0);
    return d.createBindGroup({
      layout: this.resolveLayout,
      entries: [
        { binding: 0, resource: this.acc.createView() },
        { binding: 1, resource: this.extra.createView() },
        { binding: 2, resource: this.baseTex.createView() },
        { binding: 3, resource: src.createView() },
        { binding: 4, resource: out.createView() },
        { binding: 5, resource: this.mean.createView() },
      ],
    });
  }

  private scratchTex: GPUTexture[] = [];
  private scratch(i: number) {
    if (!this.scratchTex[i])
      this.scratchTex[i] = this.d.createTexture({
        size: [this.size, this.size],
        format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
    return this.scratchTex[i];
  }

  /** Adds one photo (and its subject mask, if any) seen from `v`; both
   *  already at the size the intrinsics are scaled to. */
  photo(v: View, photo: ImageBitmap, mask: ImageBitmap | null, scale: number) {
    const d = this.d;
    const w = photo.width, h = photo.height;
    const f = this.frameFor(w, h);
    const T = GPUTextureUsage;
    const upload = (bmp: ImageBitmap) => {
      const t = d.createTexture({ size: [w, h], format: "rgba8unorm", usage: T.TEXTURE_BINDING | T.COPY_DST | T.RENDER_ATTACHMENT });
      d.queue.copyExternalImageToTexture({ source: bmp }, { texture: t }, [w, h]);
      this.frameTextures.push(t);
      return t;
    };
    const photoTex = upload(photo);
    const maskTex = mask ? upload(mask) : photoTex;

    // camera: distance range from the mesh's bounding sphere
    const { R, t } = v;
    const centre = [0, 1, 2].map((k) => -(R[k] * t[0] + R[3 + k] * t[1] + R[6 + k] * t[2]));
    const dist = Math.hypot(...centre.map((x, k) => x - this.bounds.center[k]));
    const near = Math.max(dist - this.bounds.radius * 1.1, this.bounds.radius * 1e-3), far = dist + this.bounds.radius * 1.1;
    const cam = new Float32Array(32);
    cam.set([R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0, t[0], t[1], t[2], 0]);
    cam.set([v.fx * scale, v.fy * scale, v.cx * scale, v.cy * scale, w, h, Math.ceil(w / Q), Math.ceil(h / Q)], 16);
    cam.set([this.tolerance, near, far, mask ? 1 : 0, ...centre, 1], 24);
    d.queue.writeBuffer(this.camBuf, 0, cam);

    const enc = d.createCommandEncoder();
    // depth of the mesh from this camera
    const camGroup = d.createBindGroup({ layout: this.depthPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.camBuf } }] });
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: f.depth.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }],
      depthStencilAttachment: { view: f.zbuf.createView(), depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 1 },
    });
    rp.setPipeline(this.depthPipe);
    rp.setBindGroup(0, camGroup);
    this.draw(rp);
    rp.end();
    // min-pool of the depth and feather of the subject
    const fg = d.createBindGroup({
      layout: this.filterLayout,
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: f.depth.createView() },
        { binding: 2, resource: maskTex.createView() },
        { binding: 3, resource: { buffer: f.tmp } },
        { binding: 4, resource: { buffer: f.minDepth } },
        { binding: 5, resource: { buffer: f.feather } },
      ],
    });
    for (const e of ["minH", "minV", "edtH", "edtV"] as const) {
      const cp = enc.beginComputePass();
      cp.setPipeline(this.filter[e]);
      cp.setBindGroup(0, fg);
      const [gw, gh] = e.startsWith("edt") ? [Math.ceil(w / Q), Math.ceil(h / Q)] : [w, h];
      cp.dispatchWorkgroups(Math.ceil(gw / 16), Math.ceil(gh / 16));
      cp.end();
    }
    // texels
    const tp = enc.beginRenderPass({
      colorAttachments: [this.acc, this.pass ? this.extra : this.count0].map((tex) => ({ view: tex.createView(), loadOp: "load" as const, storeOp: "store" as const })),
    });
    tp.setPipeline(this.texelPipes[this.pass]);
    tp.setBindGroup(0, this.texelBindGroup(photoTex, f));
    this.draw(tp);
    tp.end();
    d.queue.submit([enc.finish()]);
    // the textures are only needed by the commands just submitted
    for (const tex of this.frameTextures.splice(0)) tex.destroy();
  }

  /** Composes photo and vertex colours, fills the gutters and reads the
   *  texture back as RGBA bytes. */
  async finish(): Promise<{ rgba: Uint8ClampedArray; stats: BakeStats }> {
    const d = this.d, size = this.size;
    const enc = d.createCommandEncoder();
    const groups = Math.ceil(size / 16);
    let cp = enc.beginComputePass();
    cp.setPipeline(this.resolvePipes.compose);
    cp.setBindGroup(0, this.resolveBindGroup(this.acc, this.scratch(0)));
    cp.dispatchWorkgroups(groups, groups);
    cp.end();
    let cur = 0;
    for (let i = 0; i < GUTTER; i++) {
      cp = enc.beginComputePass();
      cp.setPipeline(this.resolvePipes.dilate);
      cp.setBindGroup(0, this.resolveBindGroup(this.scratch(cur), this.scratch(1 - cur)));
      cp.dispatchWorkgroups(groups, groups);
      cp.end();
      cur = 1 - cur;
    }
    const out = d.createBuffer({ size: size * size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyTextureToBuffer({ texture: this.scratch(cur) }, { buffer: out, bytesPerRow: size * 4 }, [size, size]);
    d.queue.submit([enc.finish()]);
    await out.mapAsync(GPUMapMode.READ);
    const rgba = new Uint8ClampedArray(out.getMappedRange().slice(0));
    out.unmap();
    out.destroy();
    let photo = 0, mesh = 0;
    for (let i = 3; i < rgba.length; i += 4) {
      if (rgba[i] === 255) photo++;
      if (rgba[i] >= 204) mesh++;
      rgba[i] = 255;
    }
    return { rgba, stats: { size, photoCoverage: mesh ? photo / mesh : 0 } };
  }

  destroy() {
    this.releaseFrame();
    for (const r of [this.vertices, this.indices, this.camBuf, this.acc, this.extra, this.mean, this.count0, this.baseTex, ...this.scratchTex, this.placeholderTex]) r?.destroy();
  }
}
