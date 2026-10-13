# Pipeline design notes

These notes cover the desktop pipeline (Rust backend in `src-tauri/`). The web
version follows the same rules; its differences are in [web.md](web.md).

## Why this pipeline

| Choice | Reason |
|---|---|
| VideoToolbox decoding + sharpness selection | Extracts N×K candidates and keeps the sharpest of each window, evenly spread over the orbit. Sharpness is computed in Rust with rayon, in parallel. |
| Single pinhole camera | Generated video has no lens distortion; fewer parameters make calibration more stable. |
| Circular matching | Each frame is matched with its neighbours, wrapping around to close the loop, plus long jumps (16/32/64). Much cheaper than exhaustive matching and needs no vocab tree. |
| `view_graph_calibrator` + `global_mapper` | COLMAP 4's global mapper (formerly GLOMAP) is much faster than the incremental one. The incremental mapper is only a fallback when the global one registers fewer than 60% of the cameras. |
| Boosted SIFT | Low detection threshold (0.002), domain-size pooling and affine shape, with 20 matched neighbours. On a real AI video orbiting the back of a head (smooth hair, blurred background), default SIFT registered 83/107 cameras; boosted SIFT registered 107/107 in ~110 s. With default SIFT, if fewer than 95% of the cameras register, the app automatically retries with the boosted settings. |
| ALIKED + LightGlue (optional) | Also registered 107/107 on that video, but took 16.5 min on the CPU with a higher reprojection error (1.5 px vs 0.84 px). |
| Brush as a sidecar | GPU training on Metal, without CUDA. It exports snapshots that the viewer (Spark/three.js) reloads live; the viewer only renders when something changes, so it doesn't compete with training for the GPU. |
| SSIM loss off by default | On Metal it triples the cost of each step (8.8 steps/s with it, 26.7 without). On a real video (107 frames, 960×1280), 15k steps without SSIM reached 32.6 dB in 8.6 min; with SSIM, 7k steps reached 31.8 dB in 9.3 min and 20k reached 33.2 dB in ~28 min. The Quality preset keeps SSIM. |
| Viewer angle lock | The camera orbit is limited to the range of heights the video covered (+20°), where there is data. |
| PSNR on held-out frames | Measures the video's 3D consistency: below ~25 dB, an AI video is probably not geometrically coherent. |

On a synthetic test video (960×540, 120 frames, M4 with 16 GB): frames in 0.6 s;
poses in 25 s (120/120 cameras, 0.57 px); 3k training steps in 94 s, reaching
35.7 dB PSNR.

On Windows with CUDA, COLMAP's SIFT extraction, matching and bundle adjustment
run on the GPU. Boosted SIFT (affine shape + domain-size pooling) only exists on
the CPU; COLMAP falls back on its own and matching stays on the GPU.

## Subject masking

With **Isolate the subject** on, every frame gets a subject mask and Brush trains
with a transparent background: the splat contains only the person or object,
without walls, people in the background or floaters around it.

During masked training the background is a colour drawn at random every step
(0.5 ± 0.5) instead of Brush's black ± 0.1. On black, dark semi-transparent
layers outside the silhouette cost almost nothing and turn into a ghost
silhouette around the subject (streaks on the legs and a dark "second skin" in
the test project). With a varying background that layer disappears and the
splat ends up ~15% smaller in the same training time.

| Method | Platform | How | Measured |
|---|---|---|---|
| AI (Vision) | macOS 14+ | Native system segmentation on the GPU, via the `splatforge_mask` sidecar (built by `build.rs`) | ~40 ms/frame on an M4 |
| People | Windows | MODNet (Apache-2.0) with ONNX Runtime + DirectML | 144 ms/frame on an M4 CPU; GPU not measured |
| Objects | Windows | BiRefNet-lite (MIT) with ONNX Runtime + DirectML | ~4 s/frame on an RTX 4070 Ti Super |
| RMBG | Windows, macOS (Apple Silicon) | RMBG-1.4 (BRIA AI, **non-commercial**) with ONNX Runtime + DirectML / CoreML on the GPU | ~0.4 s/frame on an M4 (+~5 s to compile for CoreML) |
| Solid colour | all | Chroma key in Rust: RGB distance to the background colour (detected from the median of the borders, or picked; brighter areas of the screen count too), tolerance, smoothing and flood fill from the borders | 120 frames in 0.09 s |

BiRefNet is slow on any GPU because its deformable convolutions turn into ~16k
tiny ONNX operations; MODNet is a simple convolutional network and, on the
frames tested (people facing towards and away from the camera), cut out just as
well. RMBG-1.4 has cleaner edges than Vision (no bright halo around hair), but
its license is non-commercial, so it is only downloaded (176 MB, via **Install
tools**) when selected. On macOS it runs on CoreML with CPU and GPU only:
compiling for the Neural Engine can hang the system's `ANECompilerService`.

On Windows the models (MODNet 26 MB, BiRefNet 224 MB) are downloaded via
**Install tools**, and `DirectML.dll` 1.15.4 ships inside the installer.

**Despill** (any method): with a green or blue background, light bounced off the
screen tints the whole subject, not just the edges. The background colour is the
median of the pixels the mask marks as background; across the subject, that
colour's channel is clamped to the larger of the other two and the removed
luminance is added back as grey (the subject doesn't get darker; yellows and
cyans are untouched). Unsaturated backgrounds or other colours are ignored. On
a green screen with a similarly coloured subject, **RMBG + despill** gives the
best result: the AI does the cutout and despill removes the green.

Masks can optionally be passed to COLMAP too (**Also use the cutout for poses**). It helps when the background changes between frames, but removes
anchor points from low-texture subjects, so it is off by default.

## Optimized export

Once the splat is trained, the **Export** panel produces a smaller version
without retraining (you can try settings and toggle **View in 3D / View
original**):

| Setting | What it does |
|---|---|
| **Compressed SOG** | PlayCanvas format (v2): lossless WebP textures with 16-bit positions, "smallest three" quaternions, scales and colours in 256-entry codebooks and the SH bands in a k-means palette of up to 64k vectors. Opens in Spark (the app's viewer), PlayCanvas and SuperSplat. |
| **Reduce** | Merges similar Gaussians (neighbours with close shape and colour) until the requested percentage is removed. Opacity preserves the "ink" (opacity × area), so surfaces don't open holes. |
| **Min opacity** | Drops nearly invisible splats. |
| **SH bands** | Cuts view-dependent shading; each degree removed shrinks the `.ply` considerably. |

It's a Rust (rayon) port of what WebGaussian does in the browser. Measured on an M4:

| Input | Output | Time |
|---|---|---|
| 111k splats, SH 1, 10.3 MB | `.sog` 2.1 MB (4.9×) | 2.5 s |
| same, −50%, opacity ≥ 2%, SH 0 | `.sog` 0.69 MB (14.9×) | 0.2 s |
| 1M splats, SH 3, 236 MB | `.sog` 18 MB (12.8×) | 15.7 s |
| same, −50% | `.ply` 118 MB | 1.2 s |

The `.sog` output was checked against the reference reader (PlayCanvas'
`splat-transform`): position error of 5×10⁻⁵ of the scene extent, rotation
below 1.2° (p99) and SH coefficients with a median error of 0.001. WebP comes
from libwebp (built from source by `libwebp-sys`), ~18% smaller than the pure
Rust encoder.
