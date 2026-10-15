<p align="center">
  <img src="assets/icon.png" width="112" alt="Gausify Orbit icon">
</p>

<h1 align="center">Gausify Orbit</h1>

<p align="center">
  Turn <b>orbit videos</b>, including AI-generated ones, into <b>Gaussian Splats</b>.<br>
  Runs 100% locally: a desktop app for macOS and Windows, and a WebGPU version that runs entirely in the browser.
</p>

<p align="center">
  <a href="https://orbit.gausify.com"><img alt="Live demo" src="https://img.shields.io/badge/demo-orbit.gausify.com-ff9a4d?logo=googlechrome&logoColor=white"></a>
  <a href="https://github.com/obraia/gausify-orbit/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/obraia/gausify-orbit?label=release&color=orange"></a>
  <a href="https://github.com/obraia/gausify-orbit/releases"><img alt="Downloads" src="https://img.shields.io/github/downloads/obraia/gausify-orbit/total?color=brightgreen"></a>
  <a href="https://github.com/obraia/gausify-orbit/actions/workflows/web.yml"><img alt="Web build" src="https://img.shields.io/github/actions/workflow/status/obraia/gausify-orbit/web.yml?branch=main&label=web%20build"></a>
  <img alt="Platforms: macOS, Windows, Web" src="https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Web-555">
  <a href="https://v2.tauri.app"><img alt="Tauri 2" src="https://img.shields.io/badge/Tauri-2-24C8D8?logo=tauri&logoColor=white"></a>
  <a href="docs/web.md"><img alt="WebGPU" src="https://img.shields.io/badge/WebGPU-in%20the%20browser-005A9C?logo=webgpu&logoColor=white"></a>
  <img alt="Languages: English, Português, Español" src="https://img.shields.io/badge/i18n-en%20%7C%20pt--BR%20%7C%20es-8A2BE2">
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-yellow"></a>
</p>

<p align="center">
  <a href="https://orbit.gausify.com"><b>Try it in the browser</b></a> ·
  <a href="#desktop-app">Desktop app</a> ·
  <a href="docs/web.md">How the web version works</a> ·
  <a href="LICENSE">MIT License</a>
</p>

<p align="center">
  <img src="assets/screenshot.png" width="900" alt="Gausify Orbit after a full run in the browser: the trained splat of a puppy in the 3D view with the camera orbit drawn above it, the SAM 2.1 cutout preview in the sidebar and the four completed stages (frames, cutout, camera poses, splat training) with their metrics">
</p>

---

Drop in a video that circles around a subject (a person, a product, a sculpture,
or a clip from a video model), and Gausify Orbit extracts the sharpest frames,
optionally cuts out the subject, recovers the camera poses, trains a 3D Gaussian
Splat with live previews and exports a compact `.sog`/`.ply` you can share
with a link. It doesn't need CUDA: training runs on Metal, Vulkan, DX12 or WebGPU.

```
video ──► Frames ──────────► Masking ──────────► Poses ─────────────► Training ───────► splat.ply / .sog
          hardware decode    (optional)          COLMAP 4 (desktop)    Brush
          + sharpest frame   Apple Vision,       or own SfM (web);     wgpu: Metal, Vulkan,
          per window         RMBG, SAM 2.1,      circular matching,    DX12, WebGPU;
                             chroma key          global mapper         live previews
```

## Features

- **Orbit-aware frame selection:** samples N×K candidates across the chosen
  range and keeps the sharpest frame of each window, evenly spread around the
  orbit. Image sequences and folders work too.
- **Subject masking:** Apple Vision (macOS), MODNet / BiRefNet (Windows),
  RMBG-1.4, SAM 2.1 video tracking with box/point prompts (web) and a chroma
  key with despill for green and blue screens. Masked training uses a random
  background, so no ghost silhouettes form around the subject, and edge colours
  are unmixed from the background, so no light fringe forms around hair.
- **Robust poses:** COLMAP 4 with boosted SIFT, circular matching with long
  jumps and the global mapper, with automatic fallbacks. The web version ships
  its own incremental SfM (AKAZE, WebGPU matching and bundle adjustment).
- **GPU training without CUDA:** [Brush](https://github.com/ArthurBrussee/brush),
  pinned to a fixed commit and patched, runs as a sidecar on desktop and
  compiled to WASM + WebGPU in the browser. A live 3D viewer
  ([Spark](https://sparkjs.dev)) reloads snapshots as training goes.
- **Optimized export:** PlayCanvas SOG v2 (lossless WebP textures, k-means SH
  palette), decimation that merges similar Gaussians, an opacity floor and
  SH band trimming. Typical output is 5–15× smaller than the raw `.ply`.
- **Mesh export (web):** a geometric fine-tune of the splat, depth maps fused
  into a sparse TSDF on the GPU, unseen holes closed, simplification and
  feature-preserving smoothing, then a UV atlas and the photos baked into its
  texture: a `.glb` that opens in the 3D view, Blender or three.js.
- **Share by link:** uploads the export to Cloudflare R2 and returns a private,
  expiring `/s/<id>` link with a QR code. It opens in a PlayCanvas viewer with
  touch controls and **VR/AR** (WebXR).
- **Retouch with ComfyUI (web):** renders the splat along new camera paths,
  restores the clip with a video model in your ComfyUI and fine-tunes the splat
  on the restored frames.
- **Project packages:** save and reopen a whole project as a `.gorbit` file,
  even halfway through a job. Both apps use the same format.
- **UI in English, Portuguese (Brazil) and Spanish.**

## Web version

Open **[orbit.gausify.com](https://orbit.gausify.com)** in a desktop Chrome or
Edge with WebGPU. The whole pipeline runs in your browser, and videos never
leave your machine. Projects are stored in the browser (OPFS), and the app
works offline after the first visit.

To run it locally:

```bash
pnpm install
pnpm web:wasm      # clones Brush at the pinned commit, applies web/brush/*.patch, builds WASM (~5 min)
pnpm web:dev       # http://localhost:1440
```

`web:wasm` needs `rustup` and `wasm-pack`. It installs a side-by-side Rust
1.99 toolchain (Brush's Burn needs ≥ 1.95) without touching `stable`.

See **[docs/web.md](docs/web.md)** for the architecture, what had to be patched
to train in the browser, and benchmarks.

## Desktop app

A Tauri 2 + React app with a Rust backend. It runs on macOS (Apple Silicon) and
Windows; on Windows, COLMAP uses CUDA when an NVIDIA GPU is present.

### macOS

```bash
brew install ffmpeg colmap
pnpm install
pnpm setup:tools   # builds the Brush sidecar and checks ffmpeg/COLMAP
pnpm tauri dev
```

### Windows

**End users:** download `Gausify Orbit_<version>_x64-setup.exe` from the
[Releases](https://github.com/obraia/gausify-orbit/releases/latest) page. It installs per user without admin rights. On
first launch, **Install tools** downloads COLMAP (CUDA build when an NVIDIA GPU
is found) and ffmpeg, verifies their SHA-256 and installs them under
`%LOCALAPPDATA%\SplatForge\tools`. The installer isn't code-signed, so
SmartScreen will ask you to confirm (**More info → Run anyway**).

**Development:** you need Node 20+, pnpm, Rust with the MSVC toolchain (Visual
Studio Build Tools, "Desktop development with C++") and WebView2.

```powershell
pnpm install
pnpm setup:tools   # builds Brush; downloads COLMAP 4.2.1 + ffmpeg (SHA-256 verified)
pnpm tauri dev
```

`setup:tools` picks the CUDA build of COLMAP when `nvidia-smi` is found
(`--cpu` forces the CPU build). COLMAP and ffmpeg installed via winget, scoop,
Chocolatey or on `PATH` are detected as well.

### About the Brush build

The desktop app and the web version share the same Brush: `setup:tools` (or
`pnpm brush:native`) builds `brush-cli` from the pinned commit with the patches
in `web/brush/`. Official Brush releases don't have the training options the
app uses. When the commit or a patch changes, `setup:tools` rebuilds
automatically, using a stamp in the `.rev` file next to the binary.

### Building installers

```bash
pnpm tauri build   # macOS: .app and .dmg · Windows: NSIS installer
```

`.github/workflows/release.yml` builds both platforms on GitHub runners and
publishes a draft release when you push a `v*` tag (or run it manually from the
Actions tab).

## Usage

1. Drop a video into the window, or drop several images or a folder.
2. If the subject "melts" near the end of the orbit, select a stable range
   with **Start here / End here**.
3. Pick a preset and click **Generate splat**.

Each video gets a project folder (`<video>_splat/` by default). To rerun only
part of the pipeline, for example with new training settings, use **Run from →
Training**, which reuses the existing frames and poses.

```
<video>_splat/
├── project.json     state (frames, metrics, result)
├── dataset/         Brush input: images/ + sparse/0/
├── masks/           subject masks (grayscale, 255 = subject)
├── dataset_alpha/   when masking: RGBA PNGs + images.bin with .png names
├── colmap/          database.db, pairs.txt, raw models
├── training/        intermediate snapshots (keeps the latest 2)
├── splat.ply        final result
└── export/          optimized exports (.sog / .ply / .glb)
```

Images dropped instead of a video are taken in natural name order (`frame2`
before `frame10`), which should follow the orbit. They're rotated according to
EXIF, resized to the preset's max side and must all share the same size, because
COLMAP and Brush use a single camera.

### Batch / command line

The same pipeline runs headless and prints JSON events, one per line:

```bash
cd src-tauri
cargo run --release --example cli -- config.json          # a JobConfig
cargo run --release --example cli -- scene <project>      # scene framing
cargo run --release --example cli -- export <project> [options.json]
```

`options.json` follows `ExportOptions`, e.g.
`{"format": "sog", "decimate": 0.5, "minOpacity": 0.02, "shDegree": 0}`.

## Documentation

| Topic | |
|---|---|
| [Pipeline design notes](docs/pipeline.md) | Why each stage works the way it does, masking methods, export format and measurements |
| [Web version](docs/web.md) | Browser architecture, Brush WASM patches, SAM 2.1, in-browser SfM, benchmarks, Cloudflare deployment |
| [Share links](docs/sharing.md) | R2 storage, the WebXR viewer, expiry and limits, self-hosting the Worker |
| [Mesh export](docs/mesh.md) | Splat → GLB: geometric fine-tune, depth fusion, extraction, hole filling, smoothing, UV and texture bake, and why each choice |
| [Retouch with ComfyUI](docs/retouch.md) | Camera paths, reference rendering, ComfyUI integration and fine-tuning |
| [Development](docs/development.md) | Mock UI without the backend, benchmark scripts, tests |

## Tests

```bash
pnpm test:rust   # Rust backend
pnpm test:web    # TypeScript (web pipeline, SfM, export, worker)
```

## Acknowledgements and third-party licenses

Gausify Orbit builds on great open-source work:
[Brush](https://github.com/ArthurBrussee/brush) (Apache-2.0),
[COLMAP](https://colmap.github.io) (BSD),
[Spark](https://sparkjs.dev) (MIT),
[PlayCanvas](https://playcanvas.com) and
[splat-transform](https://github.com/playcanvas/splat-transform) (MIT),
[ONNX Runtime](https://onnxruntime.ai) (MIT),
[OpenCV.js](https://opencv.org) (Apache-2.0),
[Mediabunny](https://mediabunny.dev) (MPL-2.0),
[three.js](https://threejs.org) (MIT) and
[ffmpeg](https://ffmpeg.org) (LGPL/GPL, downloaded at runtime, not bundled).
The SOG export is a port of the encoder in WebGaussian.

The segmentation models are downloaded on demand and keep their own licenses:

| Model | License | Used by |
|---|---|---|
| [SAM 2.1 tiny](https://github.com/facebookresearch/sam2) | Apache-2.0 | web |
| [MODNet](https://github.com/ZHKKKe/MODNet) | Apache-2.0 | Windows |
| [BiRefNet-lite](https://github.com/ZhengPeng7/BiRefNet) | MIT | Windows |
| [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) (BRIA AI) | **Non-commercial** | Windows, macOS, web (opt-in) |

RMBG-1.4 is only downloaded when you select it. Check its license before using
it in commercial work.

## License

[MIT](LICENSE) © Bryan Diniz
