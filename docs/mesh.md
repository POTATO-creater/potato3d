# Mesh export

**Export → Mesh (GLB)** turns the trained splat into a triangle mesh coloured
by the photos. Web only for now (`web/src/mesh/`); the desktop gets a Rust port
of the same steps later.

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
5. **Simplification** with meshoptimizer to the chosen triangle count; the
   mesh is rotated so Brush's vertical axis is +Y, centred and stood on y = 0;
   GLB with normals and linear vertex colours (`glb.ts`).

## Why these choices

Measured on a 154-frame orbit of a person (see the study notes in the PR):

| Choice | Alternative | Result |
|---|---|---|
| Fine-tune of the trained splat (8 min native) | Separate mesh training (36 min) | Same mesh quality in novel views; splat PSNR 28.2 vs 27.5 dB |
| Normal consistency only, 1 extra render | + depth distortion, 2 extra renders | Same quality, 10% faster per step |
| Expected depth | Median depth | Cleaner surface: the median jumps between Gaussian layers |
| meshoptimizer simplification | Isotropic remeshing | More shape detail (neckline, stilettos), less regular triangles |
| Fine-tune with the geometric losses | Plain splat | Lumpy surface, faces nearly featureless |

Things that stay hard: anything never seen by the orbit (top of the head,
soles) is a hole, thin structures thinner than ~2 voxels (stiletto heels) can
break, and hair is a smooth shell; its detail belongs in a texture.
