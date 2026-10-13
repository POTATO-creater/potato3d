// Brute-force matching of binary descriptors (AKAZE, 486 bits in 16 u32) in a
// WebGPU compute shader: for each descriptor in A, the best and second-best in
// B by Hamming distance. Pairs are processed in batches and in both
// directions, for the ratio test and the mutual check.

const WGSL = /* wgsl */ `
struct Job { offA: u32, nA: u32, offB: u32, nB: u32, out: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read> desc: array<u32>;
@group(0) @binding(1) var<storage, read> jobs: array<Job>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;
var<workgroup> tile: array<u32, 1024>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let job = jobs[wg.y];
  let q = wg.x * 64u + lid.x;
  let valid = q < job.nA;
  var qd: array<u32, 16>;
  if (valid) {
    for (var k = 0u; k < 16u; k++) { qd[k] = desc[(job.offA + q) * 16u + k]; }
  }
  var best = 0xffffu;
  var second = 0xffffu;
  var bi = 0u;
  for (var base = 0u; base < job.nB; base += 64u) {
    let j = base + lid.x;
    for (var k = 0u; k < 16u; k++) {
      tile[lid.x * 16u + k] = select(0u, desc[(job.offB + j) * 16u + k], j < job.nB);
    }
    workgroupBarrier();
    let cnt = min(64u, job.nB - base);
    if (valid) {
      for (var t = 0u; t < cnt; t++) {
        var d = 0u;
        for (var k = 0u; k < 16u; k++) { d += countOneBits(qd[k] ^ tile[t * 16u + k]); }
        if (d < best) { second = best; best = d; bi = base + t; }
        else if (d < second) { second = d; }
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    result[job.out + q * 2u] = bi;
    result[job.out + q * 2u + 1u] = best | (second << 16u);
  }
}`;

export interface PairMatches {
  a: number;
  b: number;
  /** Feature indices in A and B, matched in both directions. */
  ia: Uint32Array;
  ib: Uint32Array;
}

export class HammingMatcher {
  private constructor(
    private device: GPUDevice,
    private pipeline: GPUComputePipeline,
  ) {}

  static async create(): Promise<HammingMatcher> {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter.");
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const module = device.createShaderModule({ code: WGSL });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    return new HammingMatcher(device, pipeline);
  }

  /**
   * Matches the requested pairs. `ratio`: Lowe's test; `maxDist`: maximum
   * Hamming distance (out of 486 bits).
   */
  async match(
    descs: Uint32Array[],
    pairs: [number, number][],
    { ratio = 0.8, maxDist = 110, batch = 96 } = {},
    onProgress?: (done: number, total: number) => void,
  ): Promise<PairMatches[]> {
    const dev = this.device;
    const counts = descs.map((d) => d.length / 16);
    const offsets: number[] = [];
    let total = 0;
    for (const c of counts) {
      offsets.push(total);
      total += c;
    }
    const all = new Uint32Array(total * 16);
    descs.forEach((d, i) => all.set(d, offsets[i] * 16));
    const descBuf = dev.createBuffer({ size: all.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(descBuf, 0, all);
    const maxN = Math.max(...counts);

    const out: PairMatches[] = [];
    for (let s = 0; s < pairs.length; s += batch) {
      const chunk = pairs.slice(s, s + batch);
      // two jobs per pair: A→B and B→A
      const jobs = new Uint32Array(chunk.length * 2 * 8);
      let outLen = 0;
      const outOff: number[] = [];
      chunk.forEach(([a, b], k) => {
        for (const [x, y, slot] of [
          [a, b, 2 * k],
          [b, a, 2 * k + 1],
        ]) {
          jobs.set([offsets[x], counts[x], offsets[y], counts[y], outLen], slot * 8);
          outOff[slot] = outLen;
          outLen += counts[x] * 2;
        }
      });
      const jobBuf = dev.createBuffer({ size: jobs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(jobBuf, 0, jobs);
      const resBuf = dev.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readBuf = dev.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const bind = dev.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: descBuf } },
          { binding: 1, resource: { buffer: jobBuf } },
          { binding: 2, resource: { buffer: resBuf } },
        ],
      });
      const enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(maxN / 64), chunk.length * 2);
      pass.end();
      enc.copyBufferToBuffer(resBuf, 0, readBuf, 0, outLen * 4);
      dev.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const res = new Uint32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      [jobBuf, resBuf, readBuf].forEach((b) => b.destroy());

      chunk.forEach(([a, b], k) => {
        const ab = outOff[2 * k], ba = outOff[2 * k + 1];
        const ia: number[] = [], ib: number[] = [];
        for (let i = 0; i < counts[a]; i++) {
          const j = res[ab + 2 * i];
          const packed = res[ab + 2 * i + 1];
          const best = packed & 0xffff, second = packed >>> 16;
          if (best > maxDist || best > ratio * second) continue;
          if (res[ba + 2 * j] !== i) continue; // mutual
          ia.push(i);
          ib.push(j);
        }
        out.push({ a, b, ia: Uint32Array.from(ia), ib: Uint32Array.from(ib) });
      });
      onProgress?.(Math.min(s + batch, pairs.length), pairs.length);
    }
    descBuf.destroy();
    return out;
  }

  destroy() {
    this.device.destroy();
  }
}
