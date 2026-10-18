# Share links

After exporting, **Share** (in the export panel) uploads the
`.sog`/`.ply` to Cloudflare R2 and creates a private `/s/<id>` link with a QR
code. Anyone with the link can open the splat in a dedicated viewer
(`web/share.html`, PlayCanvas): drag/pinch on desktop and mobile, plus **VR**
(WebXR headsets such as Quest) and **AR** (Android phones) buttons when the
device supports them. The camera and the XR session are ported from the
WebGaussian player: in XR the scene becomes a ~1.4 m tabletop model in front of
the user, with thumbstick locomotion and snap turn. The link carries the video's
framing (centre, "up", first frame, lens and filmed height range), so it opens
just like **Recenter** in the app.

Unlike WebGaussian (a single isolated object), an orbit scene has a background
and large loose splats near the cameras. In the tabletop model they surrounded
the viewer's head: on Quest 3 that dropped the frame rate and washed out the
colours with a white haze. So, only during an XR session, the viewer:

- discards, in the shader (`setWorkBufferModifier`), splats outside a sphere of
  0.8 × the camera orbit radius. In the test project this removes 24% of the
  splats, but they are the largest ones (14× the average scale of those kept);
- rotates the model so the filmed side (the first frame) faces the user;
- turns on PlayCanvas' radial sorting (fewer artifacts when turning your head)
  and fixed foveation (0.5).

For comparisons on a headset, the link accepts `?crop=` (0 disables the crop),
`?foveation=`, `?xrScale=` (XR resolution) and `?sort=linear`. `?xrPreview=1`
builds the same tabletop model in a regular browser, with the camera at head
height.

The viewer uses WebGL 2, like WebGaussian. `?gpu=webgpu` switches to WebGPU,
where PlayCanvas sorts splats on the GPU (`?gpuSort=0` goes back to CPU
sorting), and a chip shows the active backend. In VR this depends on the
`XRGPUBinding` API (WebXR/WebGPU Binding), which is still experimental: on Quest
Browser 146 or newer, enable "WebXR experimental features" in `chrome://flags`.
Without it, on a device with VR/AR the viewer stays on WebGL 2 so the buttons
don't disappear, and the chip says so. Careful: in a Chrome without the binding,
`requestAdapter({ xrCompatible: true })` never resolves and locks the tab's
WebGPU, and PlayCanvas requests it by itself whenever `navigator.xr` exists.
That's why the viewer only passes `xrCompatible` to WebGPU when the binding
exists.

## Behaviour and limits

- **Private:** the id has ~71 random bits, and the page is not indexed and sends
  no Referer. Anyone with the link can view (and download) the splat.
- **Expiry:** `SHARE_TTL_HOURS` in `wrangler.jsonc` (default 6 h). The Worker
  treats expired links as missing right away, a cron job every 30 min deletes
  them from R2, and the creator can **Revoke** earlier (the app keeps the
  deletion token on the device).
- **Custom expiry:** with the `SHARE_ADMIN_TOKEN` secret set, opening the app
  with `?admin=true` and entering that key in **Share → I have an access key**
  lets you pick the expiry (6 h to 1 year, or never). The key stays on that
  device only. These links go under `p/`, outside the lifecycle rule; the cron
  job deletes them when they expire, and links without expiry stay until someone
  **Revokes** them.
- **Limits:** `SHARE_MAX_MB` (95 MiB; Cloudflare rejects bodies over 100 MB),
  only files that start like a `.sog` (ZIP) or `.ply`, and 10 uploads per minute
  per IP (`ratelimits`).
- **Server:** the page's own origin; `VITE_SHARE_ORIGIN` points `web:dev` at
  another one, which must list the page's origin in `SHARE_ORIGINS` (CORS).

## Self-hosting

One-time setup (deploying fails without the bucket):

```bash
pnpm exec wrangler r2 bucket create gausify-orbit-shares
```
```bash
pnpm exec wrangler r2 bucket lifecycle add gausify-orbit-shares expire-links s/ --expire-days 1
```

To enable custom expiry (optional; without the secret every link lasts
`SHARE_TTL_HOURS`):

```bash
pnpm exec wrangler secret put SHARE_ADMIN_TOKEN
```

The lifecycle rule is just a safety net in case the cron job stops (R2 can't
expire objects in less than a day). The CI token (`CLOUDFLARE_API_TOKEN`, "Edit
Cloudflare Workers" template) already includes R2 permissions.

To test locally: `pnpm web:build`, `pnpm exec wrangler dev` (R2 and the rate
limiter are simulated in `.wrangler/`) and open `http://localhost:8787`. To use
it from the dev server: `VITE_SHARE_ORIGIN=http://localhost:8787 pnpm web:dev`
and `wrangler dev --var SHARE_ORIGINS:http://localhost:1440`.
