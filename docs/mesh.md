# Mesh export

**Export → Mesh (GLB)** turns the trained splat into a triangle mesh textured
with the photos (or, with *Photo texture* off, coloured per vertex). The code
is in `web/src/mesh/`.

## Pipeline

1. **Geometric fine-tune** (`job.ts`): 4,000 Brush steps over the trained
   splat (written as `init.ply` into the training dataset, removed afterwards)
   with the losses in `web/brush/geom.patch`:
   - *flatten* (weight 100): the smallest scale of each Gaussian goes to ~0,
     so Gaussians become discs whose shortest axis is a normal;
   - *normal consistency* (0.05, every 2nd step, at half resolution, cropped
     to the mask's box): rendered splat normals must agree with the normals of
     the rendered depth;
   - `min-scale-factor 0`: the Mip-Splatting 3D filter compensates opacity for
     Gaussians thinner than its floor; with flat Gaussians that drives them to
     zero opacity, refine prunes them and densification stops.
   The splat used for viewing is not touched: the fine-tuned one only feeds
   the mesh.
2. **Depth maps** (brush-js `Training.renderDepth`): for every photo, at ¼ of
   its size, the expected camera depth `Σwᵢzᵢ / Σwᵢ` and the alpha, rendered by
   Brush's own rasterizer (each Gaussian's depth as a degree-0 SH colour).
   Pixels count where alpha > 0.5 and the subject mask is on.
3. **Sparse TSDF** (`tsdf.ts`): voxel = subject bounding-box diagonal / 1000
   (~6 mm on a person), truncation 3 voxels. A first pass over the depth maps
   marks the 8³-voxel blocks near the surface; the second integrates every
   depth map and the photo's colour into them on the GPU (TSDF and weight as
   two f16, colour as RGBA8: 8 bytes per voxel).
4. **Surface** (`extract.ts`, in `mesh.worker.ts`): marching tetrahedra (six
   tetrahedra per cube around its main diagonal: no case tables and a closed
   surface wherever the volume was seen twice), components under 1% of the
   largest are dropped.
5. **Holes** (`holes.ts`): what no photo saw (top of the head, soles,
   armpits) is a boundary loop. Each loop narrower than half the subject is
   closed by ear clipping, the patch is remeshed to an even edge length (half
   the final one: split, collapse, valence flips, relaxation) and its new
   vertices are placed by biharmonic fairing (`L²x = 0` with the two
   surrounding rings fixed), so the patch continues the surrounding
   curvature: a dome on a head, nearly flat soles. The observed surface does
   not move.
6. **Simplification** with meshoptimizer to the chosen triangle count, with
   `Regularize`: slightly less detail than plain QEM, but even triangles,
   which smooth with far fewer folds.
7. **Smoothing** (`smooth.ts`): bilateral filtering of the face normals
   (σᵣ 0.15, 6 iterations; neighbours across a crease barely count, so the
   neckline and the dress hem survive while bumps flatten), vertices moved to
   agree with them (10 iterations), then the folded faces (normal against
   their neighbourhood, seen as pin-prick pits) are relaxed. Thin parts, where
   a ray into the surface meets the other side within 2.5 mean edges
   (stilettos, fingers), stay frozen: smoothing would thin them until they
   break.
8. **UV atlas** (`uv.ts`, xatlas compiled to WebAssembly, in the worker):
   the mesh is cut into charts packed into one square texture (4096² from
   50k triangles, 2048² below), 4 texels of padding.
9. **Texture bake** (`bake.ts`, WebGPU, two passes over every photo at full
   resolution, up to 4096 px): each texel is a point on the mesh, coloured by
   the photos that see it:
   - visibility: the mesh rendered from the photo's camera and that depth
     min-pooled over 6 px; nothing nearer may lie close by, so an arm in
     front never paints the body behind it;
   - weight cos⁶ of the viewing angle, times a feather that fades the photo
     out over the last 40 px of its subject mask: seams between photos become
     gradients;
   - the second pass drops views more than 0.12 from the first pass's mean
     where at least three views agree (specular glints, occluders);
   - texels only seen at grazing angles (cos < 0.4) keep the TSDF vertex
     colour; chart borders are dilated 8 texels; JPEG at quality 0.9.
10. The mesh is rotated so Brush's vertical axis is +Y, centred and stood on
    y = 0; GLB with normals and the texture (`glb.ts`), or with linear vertex
    colours without one (glTF multiplies the two). The 3D view opens it.

## Why these choices

Measured on a 154-frame orbit of a person (see the study notes in the PR):

| Choice | Alternative | Result |
|---|---|---|
| Fine-tune of the trained splat (8 min) | Separate mesh training (36 min) | Same mesh quality in novel views; splat PSNR 28.2 vs 27.5 dB |
| Normal consistency only, 1 extra render | + depth distortion, 2 extra renders | Same quality, 10% faster per step |
| Expected depth | Median depth | Cleaner surface: the median jumps between Gaussian layers |
| meshoptimizer simplification (`Regularize`) | Isotropic remeshing | Runs in the browser, keeps the neckline and stilettos; a little less smooth after filtering |
| Fine-tune with the geometric losses | Plain splat | Lumpy surface, faces nearly featureless |
| Hole fill + biharmonic fairing | Screened Poisson | Poisson adds a slab under the feet and erases stilettos |
| Bilateral normal filter on the simplified mesh | Taubin on the dense mesh | Taubin thins stilettos until they break |
| Fill before simplifying, patch at ½ the final edge | Fill at the voxel size | Same shape, 7× faster (1.4 vs 10 s) |
| Photo texture | Vertex colours | Face, eyes and fabric pattern sharp; vertex colours blur at 50k triangles |
| Feather + robust mean | Best view per texel | The best view flips every few texels: marbled seams |
| Vertex colour where photos only graze | Harmonic fill of the texture | No blurry patches under the arms, no extra solve |

Things that stay hard: what the orbit never saw is a guess (a smooth dome on
the head, flat soles), structures thinner than ~2 voxels (stiletto heels) can
break in the TSDF, and hair is a smooth shell whose detail comes only from
the texture. The texture keeps the lighting of the photos (no delighting),
and exposure differences between photos are only feathered, not corrected.
