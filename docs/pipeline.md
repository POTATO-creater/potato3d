# Pipeline design notes

Why each stage works the way it does. How it runs in the browser (WASM,
WebGPU, OPFS, the Brush patches, the SfM) is in [web.md](web.md).

## Why this pipeline

| Choice | Reason |
|---|---|
| Sharpness selection | Extracts N×K candidates and keeps the sharpest of each window, evenly spread over the orbit. |
| Single pinhole camera | Generated video has no lens distortion; fewer parameters make calibration more stable. |
| Circular matching | Each frame is matched with its neighbours, wrapping around to close the loop, plus long jumps (16/32). Much cheaper than exhaustive matching and needs no vocab tree. |
| Orbit-aware SfM | Our own incremental SfM knows the frames come in order around a subject: it closes the orbit, merges tracks without conflicts on repetitive textures, and infers poses for frames it can't register. On a synthetic orbit with known poses it was more accurate than COLMAP (0.27% vs 0.73% of the radius). |
| Brush on WebGPU | GPU training without CUDA. It exports snapshots that the viewer (Spark/three.js) reloads live; the viewer only renders when something changes, so it doesn't compete with training for the GPU. |
| SSIM loss off in Preview and Balanced | With the pinned Brush it costs about 10% more time per step and adds detail; the Quality preset keeps it. (With Brush 0.3 it tripled the cost of each step, which is why it was left out at first.) |
| Viewer angle lock | The camera orbit is limited to the range of heights the video covered (+20°), where there is data. |
| PSNR on held-out frames | Measures the video's 3D consistency: below ~25 dB, an AI video is probably not geometrically coherent. |

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

| Method | How |
|---|---|
| RMBG-1.4 | BRIA AI's background removal (**non-commercial** license) on ONNX Runtime Web with WebGPU; only downloaded (176 MB) when selected. |
| SAM 2.1 | Meta's video segmentation (Apache-2.0): mark the subject with a box or points on one frame and it is tracked across the others. See [web.md](web.md). |
| Solid colour | Chroma key: RGB distance to the background colour (detected from the median of the borders, or picked; brighter areas of the screen count too), tolerance, smoothing and flood fill from the borders. |

**Despill** (any method): with a green or blue background, light bounced off the
screen tints the whole subject, not just the edges. The background colour is the
median of the pixels the mask marks as background; across the subject, that
colour's channel is clamped to the larger of the other two and the removed
luminance is added back as grey (the subject doesn't get darker; yellows and
cyans are untouched). Unsaturated backgrounds or other colours are ignored. On
a green screen with a similarly coloured subject, **RMBG + despill** gives the
best result: the AI does the cutout and despill removes the green.

**Edge decontamination** (any method, always on): a soft mask edge keeps the
photo's colour, a mix of subject and background, and Brush premultiplies it by
the alpha, so the background leaks into the training target and the splat
learns a light fringe around hair and outlines. Before despill, the colour of
each partial-alpha pixel is unmixed from the background estimated nearby
(`web/src/pipeline/decontam.ts`). On a white-wall video it removed the fringe
entirely (soft-edge brightness relative to the interior next to it: 1.25 →
1.00) for ~0.2 dB of interior PSNR against the original photos.

Masks can optionally be used for the poses too (**Also use the cutout for
poses**). It helps when the background changes between frames, but removes
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
| **Mesh (GLB)** | A textured triangle mesh instead of splats; see [mesh.md](mesh.md). |

It follows what WebGaussian does in the browser. Typical sizes:

| Input | Output |
|---|---|
| 111k splats, SH 1, 10.3 MB | `.sog` 2.1 MB (4.9×) |
| same, −50%, opacity ≥ 2%, SH 0 | `.sog` 0.69 MB (14.9×) |
| 1M splats, SH 3, 236 MB | `.sog` 18 MB (12.8×) |

The `.sog` output was checked against the reference reader (PlayCanvas'
`splat-transform`): position error of 5×10⁻⁵ of the scene extent, rotation
below 1.2° (p99) and SH coefficients with a median error of 0.001. WebP comes
from libwebp compiled to WASM.
