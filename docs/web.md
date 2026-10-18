# Web version

`web/` runs the entire pipeline in the browser with the **same UI** as the
desktop app: `web/src/main.tsx` mounts `src/App.tsx` and the components in
`src/components/`. Everything that differs between the two lives behind the
`Host` interface (`src/lib/api.ts`):

- **Desktop** (`src/lib/tauri.ts`): Rust backend commands through Tauri,
  native dialogs and "Show in Finder".
- **Web** (`web/src/host/`): the same commands (`startJob`, `loadProject`,
  `exportSplat`…) and the same events (`JobEvent`), implemented with WebGPU,
  WASM and OPFS. `job.ts` follows the rules of `pipeline.rs` (stages, resuming,
  `project.json`).

A new UI feature shows up in both versions; anything platform-specific checks
`tools.os` (`"web"` in the browser), just like macOS and Windows.

```bash
pnpm web:wasm      # clones Brush at the pinned commit, applies web/brush/*.patch and builds (~5 min)
pnpm web:dev       # http://localhost:1440
```

`web:wasm` needs `rustup` and `wasm-pack`; it installs a side-by-side Rust 1.99
toolchain (Brush's Burn needs ≥ 1.95) without touching `stable`. Browser:
desktop Chrome or Edge with WebGPU and the `subgroups` feature.

## Differences from the desktop app

- **Projects** live in browser storage (OPFS) under `/projects/<video name>`,
  with the same layout and the same `project.json`; the 3 most recent are kept.
  Dropping the same video again brings the project back with its finished
  stages. **Import…** copies a desktop project folder or a COLMAP dataset
  (`images/` + `sparse/0`).
- **Files:** "Show in Finder" becomes **Download**.
- **Masking:** RMBG-1.4, SAM 2.1 (web only, for now) and solid colour. RMBG
  (176 MB, non-commercial license) is downloaded through the same first-use
  prompt as on desktop, with its SHA-256 verified. SAM 2.1 (Apache-2.0) is
  downloaded on first use: 84 MB in fp16 where the GPU has `shader-f16` (Chrome
  on Mac does), 165 MB in fp32 elsewhere. With it, the player gets a ⌖ button to
  mark the subject (on the video or on dropped images): drag for a box, click
  for a subject point, Alt+click (or right click) for a background point and
  Shift+click to remove a point or the box. On a video you can only mark inside
  the selected range; if the range changes and the mark falls outside it, the
  panel warns and the mark is ignored. Without a mark, the object at the centre
  of the first frame is used.
  The masking panel shows a **preview** on the marked frame (or the start of
  the range) with the background replaced by a checkerboard. It works for RMBG,
  SAM 2.1 and solid colour and follows the marks: the encoder runs once per
  frame and each new point only runs the decoder. The preview can be hidden,
  which frees the GPU before a job.
- **Poses:** own SfM (below); the COLMAP options (features, camera model, flat
  orbit) are hidden.
- **Export:** the same as desktop (PLY or SOG, pruning, SH band trimming and
  decimation), ported to TypeScript in `web/src/export/` and running in a
  worker. Lossless WebP comes from splat-transform's libwebp WASM build (MIT,
  `web/src/export/webp/`), because canvas premultiplies alpha.
- **Video fps:** read from the MP4/MOV container, like ffprobe; for other
  formats it is measured during playback.
- **Reduce haze** (advanced training settings, off by default): exists only
  here, see [below](#reduce-haze).

## What it took to train in the browser (and fast)

- **Brush commit:** pinned before #546, which brings the `project_forward`
  kernel to 12 storage buffers; Chrome only exposes 10 per stage on Metal, so
  training wouldn't run on a Mac.
- **`enable subgroups;`:** naga's WGSL writer (wgpu 30) uses subgroup functions
  without declaring the extension, and Chrome rejects the shaders.
  `web/src/brush.ts` prepends the directive before handing the WGSL to the
  browser.
- **`web/brush/brush-js.patch`:**
  - Static constructors run only once. Without this, wasm-ld treats the module
    as a "command" and wraps **every** export (even `malloc`) to rerun CubeCL's
    thousands of `inventory` registrations on each JS→WASM call. Training was
    ~25× slower (4 steps/s).
  - CubeCL autotune set to `Full`: at other levels Burn measures the GPU's peak
    throughput with a blocking `sync`, which panics in WASM on the first reduce.
  - `trainSteps` doesn't poll an already finished stream again (panic "async fn
    resumed after completion" at the end of training).
  - `Training.exportPly()`: the same export as native Brush (built-in scale
    floor, dataset units restored).
- **`web/brush/deg.patch` (experimental, everything off by default):** two
  ideas from DeG (the TripoSplat paper).
  - Opacity and volume regularization (`opacBinaryWeight`, `opacSolidWeight`,
    `volumeRegWeight` in the training options). On the masked test orbit,
    `opacBinaryWeight: 0.01` took semi-transparent Gaussians from 71% to 14%
    at −0.2 dB on the test frames; `volumeRegWeight: 1` made them ~3× smaller
    and 46% fewer, at −0.4 dB. Visually the baseline looked better (softer
    edges), so no preset enables these options.
  - Exact contribution of each Gaussian to the view's L1 loss (how much the
    loss rises if it is removed), in a sibling kernel of the rasterizer's
    backward pass; `contribPruneMinViews > 0` prunes the ones with a negative
    balance (`contribPruneRatio` and `contribPruneStart` tune it). It didn't
    improve the look in tests: the haze comes from Gaussians that do help the
    frames.
  - Native test: `cargo +1.99 test -p brush-render --lib
    contribution_matches_removal` (in the `web/.brush` clone). `pnpm web:wasm`
    recreates the clone from scratch, so regenerate the patch (`git diff`
    against the commit with brush-js.patch applied) before building.

## Stages in the browser

1. **Frames:** the browser decodes the video (one seek per candidate). Uniform
   candidates across the range (`frames × candidates`, at most the video's
   fps), a scale that fits the max side without upscaling, and the sharpest of
   each window (variance of the Laplacian on a 640 px thumbnail, Rec.709 luma),
   as in `frames.rs`. Each seek goes half a frame ahead to land on the frame
   closest to the timestamp, like ffmpeg's `fps` filter.
2. **Masking:**
   - **RMBG-1.4:** ONNX Runtime Web on WebGPU, with the same pre/post-processing
     as `masks.rs` and the model from the same pinned URL as the installer.
     Vision is macOS-only, BiRefNet is slow even on a dedicated GPU and MODNet
     (people) was left out by design.
   - **SAM 2.1 tiny** (`sam2.ts`): segments the marked frame and propagates the
     mask forwards and then backwards using the model's memory, so the cutout
     doesn't change from frame to frame the way RMBG's does, which decides each
     frame on its own. The four graphs are the video export from
     `jax-image-tools/sam21-tiny-video-onnx` or, with `shader-f16`, its fp16
     version published at
     [`obraia/sam2.1-tiny-video-onnx-fp16`](https://huggingface.co/obraia/sam2.1-tiny-video-onnx-fp16)
     (commit and SHA-256 pinned for both). The fp16 build comes from
     `scripts/sam2-export/export_fp16.py`, which replaces the encoder's two
     `If` nodes with the branch that always runs and converts with ONNX
     Runtime's converter (positions kept in fp32); the script reproduces the
     files byte for byte. The memory bank (conditioning frames, the last 6
     tracked frames and up to 16 object pointers with temporal encoding) is
     TypeScript and mirrors `Sam2VideoModel` from Transformers 4.57. The three
     parameters it uses outside the graphs are in `sam2-params.bin` (68 KB,
     extracted from `facebook/sam2.1-hiera-tiny`). The exported encoder
     declares a wrong shape in an `If` branch that never runs, and ORT Web 1.30
     rejects the model because of it: `fixVisionShape` patches that byte in
     memory (5 → 4); the fp16 build already has no such `If`.
     `scripts/web-bench/sam2.mjs` runs the tracker over a folder of frames and
     saves the logits (`--variant fp16|fp32`).
   - **Solid colour** and **despill:** exact ports of the Rust code (tests in
     `web/src/pipeline/*.test.ts`, `pnpm test:web`, with the cases from
     `masks.rs`).
   - **Edge decontamination:** port of `decontam.rs` (`web/src/pipeline/decontam.ts`,
     same tests), ~0.3 s per 4K frame.
   - `dataset_alpha`: RGBA PNGs with straight alpha (own encoder, since canvas
     premultiplies alpha) and `images.bin` with `.png` names. **Also use the cutout for poses** drops features where the mask is 0.
3. **Poses:** SfM without COLMAP (`web/src/sfm/`):
   1. **Features:** AKAZE from OpenCV.js (`@techstark/opencv-js` 4.12, which
      ships neither SIFT nor the epipolar geometry functions) in Web Workers.
   2. **Matching:** brute-force Hamming distance in a WebGPU compute shader,
      with ratio test and mutual check; same pairs as the desktop app (preset
      window + 16/32/64 jumps, closed orbit).
   3. **Verification:** fundamental matrix by RANSAC (normalized 8-point).
   4. **Incremental reconstruction** in a worker: union-find tracks, initial
      pair chosen by parallax (≥ 2°) and number of points, with up to 3 attempts
      in different parts of the orbit if the first registers fewer than 90% of
      the cameras, PnP starting from the neighbouring frame's pose (with DLT +
      RANSAC as backup), triangulation, local BA every frame and periodic
      global BA (Levenberg–Marquardt with Schur complement, Huber loss, shared
      focal length). At the end, BA with a decreasing threshold recovers the
      observations that close the orbit.
4. **Training:** Brush in WASM + WebGPU, with previews in the viewer every
   "Preview every" steps.

## Mesh export

**Export → Mesh (GLB)** exists only on the web for now. The pipeline, its
parameters and the measurements behind them are in [mesh.md](mesh.md); the
Brush side is `web/brush/geom.patch` (geometric losses and
`Training.renderDepth`). UVs come from xatlas (npm `xatlas-wasm`, MIT), the
texture bake is plain WebGPU (`web/src/mesh/bake.ts`).

## Reduce haze

**Reduce haze** adds the depth distortion loss of 2DGS
(`depth-distortion-weight` 10, from `web/brush/geom.patch`), which pulls each
ray's weight onto a single surface. It only runs in masked training, every 2
steps from 25% of the run, rendered inside the mask's box
(`web/src/pipeline/train-config.ts`). Without masks it is never applied: fog
sheets next to the cameras dominate the term and training collapses. The
desktop app doesn't have it, because its Brush sidecar is built without that
option: the toggle only shows on hosts with `depthDistortion`, and the desktop
drops the field from web packages.

It is meant for AI-generated videos. Their frames don't quite agree with each
other, and that leaves semi-transparent layers around the subject that show up
from new angles. It costs about 25% more training time.

On a clean synthetic orbit, off is the default for a reason. The test used
`synth-orbit.py --green` (90 frames at 960×720), chroma masks, the Preview
preset and native Brush with 2 seeds:
- The off-surface weight in novel views dropped from 3.79% to 1.55%. That is
  the share of each subject pixel's weight more than 3% of the depth away from
  the pixel's median surface.
- There those layers were real detail, so PSNR dropped from 30.27 to 29.51 dB.

Other measurements:
- Every 2 steps instead of every step costs about half the extra time and
  removes slightly less haze. Half-resolution renders kept the haze reduction
  but saved less time.
- Weight 100 hardened the silhouettes.
- In the browser (headless Chrome on an M4, the same synthetic orbit,
  Preview), training went from 2 min 44 s to 3 min 29 s (+27%). PSNR went from
  30.25 to 29.69 dB, in line with native Brush on the same dataset
  (30.26 → 29.60).

## Benchmarks

On a synthetic green-screen video (`synth-orbit.py --green`: 120 frames at
960×720, 24 fps, green spill on the objects), Preview preset (90 frames), in
headless Chrome on an M4:

| Stage | Time | Result |
|---|---|---|
| Frames | 3.1 s | 90 of 120 candidates |
| Masking, solid colour | 2.4 s | IoU 0.91 against ground-truth masks (min 0.83) |
| Masking, RMBG-1.4 (WebGPU) | 32 s (+12 s on first use: download and verification) | IoU 0.996 (min 0.99), ~0.36 s/frame |
| Poses | 29–31 s | 90/90 cameras, position 0.15% of the radius, rotation 0.12°, focal −0.4% |
| Transparent PNGs | 4.5 s | subject pixels with dominant green: 16% → 0% |
| Training (3k steps, solid colour) | — | 36.7 dB PSNR, 0.4% of visible Gaussians green |

SAM 2.1 (WebGPU, M4): ~0.75 s/frame in fp16 and ~1.1 s in fp32. In fp32 that is
~0.35 s for the encoder and up to ~0.65 s for attention over a full memory
(~28k tokens). fp16 masks have IoU ≥ 0.993 per frame against fp32, with no
change against ground truth. The output matches Transformers' PyTorch
propagation (mean per-frame IoU: 0.997 in fp32 and 0.996 in fp16). On an orbit
with a cluttered scene (a torus on a pedestal with a cone, cube and sphere
around it), RMBG's coverage varied between 10% and 64% (it took the whole floor
in some frames; IoU between neighbouring frames as low as 0.16). SAM 2.1,
prompted with a box, stayed on the torus + pedestal in every frame. It only
tracks what was marked: an object hidden in the marked frame (the cube behind
the sphere in `synth-orbit.py --green`) is left out. Also, in the tracker's
multi-mask mode, a multi-part subject can lose one of its parts when its
appearance changes (PyTorch does the same).

The chroma key doesn't reach IoU 1 because the cube has blue-green patches that
touch its edge and get removed together with the screen (the desktop app does
the same); RMBG + despill keeps them. In a hidden panel or tab the browser
throttles seeks and frame extraction gets ~10× slower.

Poses measured on a synthetic 120-frame 960×720 orbit with known ground-truth
poses (`synth-orbit.py`: procedural texture, floor and back wall):

| | Time | Cameras | Points | Focal | Position (median) | Rotation (median) | Training PSNR |
|---|---|---|---|---|---|---|---|
| Browser | 37 s | 120/120 | 14k | +0.22% | 0.27% of radius | 0.085° | 39.7 dB |
| Native COLMAP 4.2 (SIFT, global) | 14 s | 120/120 | 2k | +0.07% | 0.73% of radius | 0.19° | 36.2 dB |
| Ground-truth poses | — | — | — | — | — | — | 41.2 dB |

PSNR: Brush web, 5k steps at 960 px, SH 1, 8 evaluation views. Training runs at
~40 steps/s on this dataset (about a third of native Brush).

To reproduce (`scripts/web-bench/`, with `WEB_FS_ANY=1 pnpm web:dev` running):

```bash
python3 scripts/web-bench/synth-orbit.py /tmp/orbit 120 960 720
```
```bash
ffmpeg -framerate 24 -i /tmp/orbit/images/%05d.jpg -c:v libx264 -pix_fmt yuv420p /tmp/orbit.mp4
```
```bash
node scripts/web-bench/headless.mjs poses "http://localhost:1440/?video=/@fs/tmp/orbit.mp4&dataset=/@fs/tmp/orbit" --out /tmp/orbit-web
```
```bash
python3 scripts/web-bench/eval-poses.py /tmp/orbit/gt.json /tmp/orbit-web/sparse/0
```
```bash
node scripts/web-bench/headless.mjs train "http://localhost:1440/?video=/@fs/tmp/orbit.mp4&dataset=/@fs/tmp/orbit-web" --steps 5000 --max-res 960
```

`headless.mjs` drives the UI in headless Chrome (no hidden-tab throttling).
`?video=` opens the video as if it had been dropped on the window; `?dataset=`
starts that video's project from a COLMAP dataset (images and poses), to train
or recompute poses from the right stage. Modes: `video` (video to splat, with
`--mask chroma|rmbg`), `poses`, `train` and `profile` (CPU profile of training;
with `pnpm web:wasm --profiling` it includes the WASM function names).

The browser SfM hasn't been tested on real AI videos yet (blur, low texture,
moving elements), where boosted SIFT and COLMAP's global mapper make a
difference.

## Deploying to Cloudflare

The web version is published as **static assets on Cloudflare Workers**
(`wrangler.jsonc`), which don't wake the Worker and aren't billed. The Worker
code (`worker/index.ts`) only runs for share links (`/api/*` and `/s/*`, see
[sharing.md](sharing.md)). The address is `<name>.<subdomain>.workers.dev`; for
a custom domain, see the `routes` comment in `wrangler.jsonc`.

- **Automatic:** `.github/workflows/web.yml` checks every PR (types, tests,
  build and size limit) and deploys on every push to `main`. It needs the
  `CLOUDFLARE_API_TOKEN` repository secret (a token from the "Edit Cloudflare
  Workers" template). The Brush WASM build runs in CI and stays cached until the
  pinned commit or the patches change.
- **Manual:** `pnpm web:deploy` (build + `wrangler deploy`, using the account
  from `wrangler login`).

Constraints imposed by Workers:

- **25 MiB per file:** ONNX Runtime's `.wasm` (~27 MB) is shipped gzipped
  (~7 MB) and decompressed in the browser (`web/ort-wasm-plugin.ts`).
- **`web/public/_headers`:** immutable cache on `/assets`, `no-cache` on
  `sw.js` and `Referrer-Policy: same-origin`. Hugging Face answers 404 without
  CORS when the Referer is a `*.workers.dev` page, which would break the RMBG
  download; `segment.ts` also fetches the model without a Referer.
- **Offline:** the service worker (`web/sw-plugin.ts`) caches the whole build
  on the first visit and announces new versions. It doesn't intercept `/s/*` or
  `/api/*`.
