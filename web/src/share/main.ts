// Shared-link viewer (/s/<id>): downloads the splat from the Worker
// (worker/index.ts) and shows it with PlayCanvas, which runs on desktop,
// mobile, WebXR headsets (VR) and AR-capable phones. Independent of the app:
// it loads neither React nor the pipeline.

import "./share.css";
import {
  Application,
  Asset,
  Color,
  createGraphicsDevice,
  DEVICETYPE_WEBGL2,
  DEVICETYPE_WEBGPU,
  Entity,
  FILLMODE_NONE,
  GSPLAT_RENDERER_RASTER_CPU_SORT,
  Quat,
  RESOLUTION_AUTO,
  Vec3,
} from "playcanvas";
import { CATALOGS, initialLocale } from "../../../src/i18n/catalogs";
import { fmtBytes } from "../../../src/lib/format";
import type { ShareView } from "../../../src/lib/share";
import { OrbitCamera } from "./orbit-camera";
import { XrSession, type XrOptions } from "./xr-session";

const locale = initialLocale();
const t = CATALOGS[locale].shareViewer;
document.documentElement.lang = locale;
document.title = `${t.title} · Gausify Orbit`;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// /s/<id> in production; in `pnpm web:dev` (no Worker), /share.html?id=<id>
// with VITE_SHARE_ORIGIN pointing at a `wrangler dev`.
const id = location.pathname.match(/^\/s\/([0-9A-Za-z]+)/)?.[1] ?? new URLSearchParams(location.search).get("id") ?? "";
const API = `${import.meta.env.VITE_SHARE_ORIGIN ?? ""}/api/share/${id}`;

// XR settings, overridable from the URL to compare on the headset itself
// (e.g. /s/<id>?crop=0&foveation=0):
//   crop       crop radius around the subject, in units of the camera orbit
//              radius (0 shows the whole scene)
//   foveation  fixed foveation, 0–1
//   xrScale    XR resolution (1 = the device's recommended one)
//   sort       "radial" or "linear"
//   xrPreview  1 builds the XR diorama in a regular browser, with the camera at
//              head height (to check without a headset)
//   gpu        "webgpu" renders with WebGPU instead of WebGL 2 (see `createDevice`)
//   gpuSort    0 keeps splat sorting on the CPU even on WebGPU
const query = new URLSearchParams(location.search);
const param = (key: string, fallback: number) => {
  const v = Number(query.get(key));
  return query.has(key) && Number.isFinite(v) && v >= 0 ? v : fallback;
};
const XR_OPTIONS: XrOptions = {
  framebufferScale: param("xrScale", 1) || 1,
  foveation: Math.min(1, param("foveation", 0.5)),
  radialSort: query.get("sort") !== "linear",
};
const XR_CROP = param("crop", 0.8);
const WANT_WEBGPU = query.get("gpu") === "webgpu";

/** WebGL 2 by default, like WebGaussian. With `?gpu=webgpu`, WebGPU: in XR that
 *  depends on `XRGPUBinding` (WebXR/WebGPU Binding), still experimental (in
 *  Quest Browser, behind the "WebXR experimental features" flag). Without it,
 *  on a VR/AR-capable device it stays on WebGL 2 so the buttons don't
 *  disappear. The active backend shows in a chip when the parameter is set. */
async function createDevice(canvas: HTMLCanvasElement) {
  const xr = "xr" in navigator;
  const binding = "XRGPUBinding" in globalThis;
  const immersive = async () => {
    const supported = (mode: XRSessionMode) => navigator.xr!.isSessionSupported(mode).catch(() => false);
    return xr && ((await supported("immersive-vr")) || (await supported("immersive-ar")));
  };
  const webgpu = WANT_WEBGPU && "gpu" in navigator && (binding || !(await immersive()));
  const device = await createGraphicsDevice(canvas, {
    deviceTypes: webgpu ? [DEVICETYPE_WEBGPU, DEVICETYPE_WEBGL2] : [DEVICETYPE_WEBGL2],
    antialias: false,
    // Without the binding, `requestAdapter({ xrCompatible: true })` never
    // resolves in Chrome (and blocks later requests); PlayCanvas turns the
    // option on by itself when `navigator.xr` exists. On WebGL it maps to makeXRCompatible.
    xrCompatible: webgpu ? binding : xr,
    // Passed through to WebGL (not in the type): opaque canvas, as in `new Application`.
    ...{ alpha: false },
  });
  // Without WebGPU or WebGL 2, PlayCanvas returns a "null" device that doesn't draw.
  if (!device.isWebGPU && !device.isWebGL2) throw new Error("WebGL 2 unavailable");
  if (query.has("gpu")) {
    const chip = $("backend");
    chip.hidden = false;
    chip.textContent = device.isWebGPU ? "WebGPU" : WANT_WEBGPU ? "WebGL 2 (no XRGPUBinding)" : "WebGL 2";
  }
  return device;
}

interface Meta {
  name: string;
  format: "sog" | "ply";
  size: number;
  /** `null`: never expires. */
  expiresAt: number | null;
  view: ShareView | null;
}

/** The link expired, was revoked or never existed. */
class Gone extends Error {}

const FOV = 50;
const UP = new Vec3(0, 1, 0);

function setStatus(text: string, fraction?: number, error = false) {
  $("status").hidden = false;
  $("status").classList.toggle("error", error);
  $("status-text").textContent = text;
  $("progress").parentElement!.hidden = fraction === undefined;
  if (fraction !== undefined) $("progress").style.width = `${Math.round(fraction * 100)}%`;
}

function showGone() {
  $("status").hidden = true;
  $("gone").hidden = false;
  $("gone-title").textContent = t.goneTitle;
  $("gone-body").textContent = t.goneBody;
  $("make-yours").textContent = t.makeYours;
}

function fmtLeft(ms: number) {
  const min = Math.max(1, Math.ceil(ms / 60_000));
  const h = Math.floor(min / 60);
  const d = Math.floor(h / 24);
  if (d) return `${d} d ${h % 24} h`;
  return h ? `${h} h ${min % 60} min` : `${min} min`;
}

function showExpiry(expiresAt: number) {
  const chip = $("expires");
  chip.hidden = false;
  const tick = () => {
    const left = expiresAt - Date.now();
    chip.textContent = left > 0 ? t.expiresIn(fmtLeft(left)) : t.expired;
    chip.classList.toggle("expired", left <= 0);
  };
  tick();
  setInterval(tick, 30_000);
}

async function download(url: string, size: number, onProgress: (f: number) => void): Promise<Blob> {
  const res = await fetch(url);
  if (res.status === 404) throw new Gone();
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
  const reader = res.body.getReader();
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(1, got / size));
  }
  return new Blob(chunks);
}

/** Rotation taking unit vector `a` to `b`. */
function rotationBetween(a: Vec3, b: Vec3): Quat {
  const d = a.dot(b);
  if (d < -0.999999) {
    // Opposite: half turn around any perpendicular axis.
    const axis = new Vec3().cross(new Vec3(1, 0, 0), a);
    if (axis.lengthSq() < 1e-6) axis.cross(new Vec3(0, 0, 1), a);
    return new Quat().setFromAxisAngle(axis.normalize(), 180);
  }
  const c = new Vec3().cross(a, b);
  return new Quat(c.x, c.y, c.z, 1 + d).normalize();
}

async function createViewer(view: ShareView | null) {
  const canvas = $<HTMLCanvasElement>("canvas");
  const app = new Application(canvas, { graphicsDevice: await createDevice(canvas) });
  // On WebGPU, PlayCanvas sorts splats on the GPU (renderer still
  // experimental); `?gpuSort=0` falls back to CPU sorting, as on WebGL.
  if (app.graphicsDevice.isWebGPU && query.get("gpuSort") === "0") {
    app.scene.gsplat.renderer = GSPLAT_RENDERER_RASTER_CPU_SORT;
  }
  app.setCanvasFillMode(FILLMODE_NONE);
  app.setCanvasResolution(RESOLUTION_AUTO);
  app.graphicsDevice.maxPixelRatio = Math.min(window.devicePixelRatio, 2);

  const camera = new Entity("camera", app);
  camera.addComponent("camera", { clearColor: new Color(0.051, 0.055, 0.063), fov: FOV });
  // XR anchors: the headset pose is composed with `rig`, and the splat lives
  // under `stage` (see xr-session.ts). Outside XR, both stay at identity.
  const rig = new Entity("xr-rig", app);
  rig.addChild(camera);
  app.root.addChild(rig);
  const stage = new Entity("xr-stage", app);
  app.root.addChild(stage);
  // Maps splat (COLMAP) coordinates to the world: orbit center at the origin
  // and the cameras' "up" along +Y, like the app's viewer.
  const root = new Entity("splat-root", app);
  stage.addChild(root);
  const q = view ? rotationBetween(new Vec3(...view.up).normalize(), UP) : new Quat();
  if (view) {
    const c = q.transformVector(new Vec3(...view.center));
    root.setLocalRotation(q);
    root.setLocalPosition(-c.x, -c.y, -c.z);
  }

  const orbit = new OrbitCamera(camera, canvas);
  const xr = new XrSession(app, camera, rig, stage, XR_OPTIONS, {
    onAvailability(vr, ar) {
      $("vr").hidden = !vr;
      $("ar").hidden = !ar;
    },
    onSessionChange(active) {
      $("tools").hidden = active;
    },
  });

  // Framing without the video (link without poses): the splat's bounding box.
  let box: { center: Vec3; radius: number } | null = null;

  /** Returns to the video's first-frame viewpoint, like the app's "Recenter". */
  function recenter() {
    if (view) {
      const first = q.transformVector(new Vec3(...view.camera).sub(new Vec3(...view.center)));
      // This lens is fixed; the video's may be much narrower (telephoto).
      // Move in until the video frame fits the screen snugly.
      let k = 1.35;
      if (view.fovY) {
        const v = Math.tan((FOV * Math.PI) / 360);
        const tv = Math.tan(view.fovY / 2);
        const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
        k = Math.max(tv / v, (tv * (view.aspect ?? 1)) / (v * aspect));
      }
      orbit.lookFrom(first.mulScalar(k), Vec3.ZERO);
    } else if (box) {
      const { center, radius } = box;
      orbit.lookFrom(new Vec3(0, radius * 0.5, radius * 1.8).add(center), center);
    }
  }

  if (view) {
    const r = view.radius;
    camera.camera!.nearClip = Math.max(r / 500, 0.001);
    camera.camera!.farClip = r * 100;
    orbit.setDistanceRange(r * 0.05, r * 10);
    if (view.polar) {
      // Polar angle θ from "up" ↔ pitch θ − 90°.
      const deg = (x: number) => (x * 180) / Math.PI;
      orbit.setPitchRange(deg(view.polar[0]) - 90, deg(view.polar[1]) - 90);
    }
    // The XR diorama is the crop sphere (the subject with some margin, inside
    // the camera ring), rotated so the first frame faces the headset wearer
    // (along +Z).
    const first = q.transformVector(new Vec3(...view.camera).sub(new Vec3(...view.center)));
    const crop = XR_CROP * r;
    xr.setContent({
      center: Vec3.ZERO,
      radius: crop || r,
      yaw: (-Math.atan2(first.x, first.z) * 180) / Math.PI,
      crop,
    });
    recenter();
  }

  new ResizeObserver(() => app.resizeCanvas()).observe(canvas);
  app.on("update", (dt: number) => (xr.active ? xr.update(dt) : orbit.update(dt)));
  app.start();

  async function load(blob: Blob, format: Meta["format"]) {
    const url = URL.createObjectURL(blob);
    // `filename` picks the parser (.sog or .ply) for a blob: URL.
    const asset = new Asset("splat", "gsplat", { url, filename: `scene.${format}` });
    try {
      await new Promise<void>((resolve, reject) => {
        asset.once("load", () => resolve());
        asset.once("error", (e: unknown) => reject(new Error(String(e))));
        app.assets.add(asset);
        app.assets.load(asset);
      });
    } finally {
      URL.revokeObjectURL(url);
    }
    const splat = new Entity("splat", app);
    splat.addComponent("gsplat", { asset });
    root.addChild(splat);
    xr.setSplat(splat);
    if (query.get("xrPreview") === "1") {
      const { head, target } = xr.preview();
      camera.camera!.nearClip = 0.01;
      camera.camera!.farClip = 1000;
      orbit.setDistanceRange(0.05, 100);
      orbit.setPitchRange(-89, 89);
      orbit.lookFrom(head, target);
    }
    if (!view) {
      const aabb = splat.gsplat?.instance?.meshInstance?.aabb;
      if (aabb) {
        box = { center: aabb.center.clone(), radius: aabb.halfExtents.length() || 1 };
        xr.setContent({ ...box, yaw: 0, crop: 0 });
        recenter();
      }
    }
  }

  return { app, orbit, xr, recenter, load };
}

function wireTools(v: Awaited<ReturnType<typeof createViewer>>) {
  const spin = $("spin");
  spin.textContent = t.spin;
  spin.onclick = () => {
    v.orbit.spin = !v.orbit.spin;
    spin.classList.toggle("on", v.orbit.spin);
  };
  $("recenter").textContent = t.recenter;
  $("recenter").onclick = () => v.recenter();

  const fs = $("fullscreen");
  fs.textContent = t.fullscreen;
  // iPhone has no element fullscreen.
  fs.hidden = !document.fullscreenEnabled;
  fs.onclick = () =>
    document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen().catch(() => {});

  // XR must be entered inside the click handler (user gesture).
  $("vr").textContent = t.vr;
  $("vr").title = t.vrTitle;
  $("vr").onclick = () => v.xr.start("vr");
  $("ar").textContent = t.ar;
  $("ar").title = t.arTitle;
  $("ar").onclick = () => v.xr.start("ar");
  $("tools").hidden = false;

  const hint = $("hint");
  hint.textContent = matchMedia("(pointer: coarse)").matches ? t.hintTouch : t.hintMouse;
  hint.hidden = false;
  const hide = () => hint.classList.add("faded");
  setTimeout(hide, 8000);
  v.orbit.onInteract = hide;
}

async function main() {
  if (!id) return showGone();
  setStatus(t.opening);
  const res = await fetch(API);
  if (res.status === 404) return showGone();
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const meta = (await res.json()) as Meta;
  if (meta.expiresAt !== null) showExpiry(meta.expiresAt);

  const viewer = await createViewer(meta.view);
  const size = fmtBytes(meta.size);
  const blob = await download(`${API}/file`, meta.size, (f) => setStatus(t.loading(Math.round(f * 100), size), f));
  setStatus(t.preparing, 1);
  await viewer.load(blob, meta.format);
  $("status").hidden = true;
  wireTools(viewer);
  $("canvas").focus({ preventScroll: true });
}

main().catch((e) => {
  if (e instanceof Gone) return showGone();
  console.error(e);
  setStatus(t.loadError(e instanceof Error ? e.message : String(e)), undefined, true);
});
