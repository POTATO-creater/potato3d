// Link sharing: uploads the export to the web version's Worker
// (worker/index.ts), which keeps it in R2 for a few hours and returns a
// private link to the viewer (web/share.html). The host says how to read the
// file and where the server is.

import type { SceneFrame } from "./api";

type Vec3 = [number, number, number];

/** Framing for the link viewer, in splat coordinates: the same "up", center
 *  and initial viewpoint as the app's viewer. */
export interface ShareView {
  center: Vec3;
  up: Vec3;
  /** Camera of the video's first frame. */
  camera: Vec3;
  /** Mean camera distance to the center. */
  radius: number;
  /** Video lens (vertical field of view in radians and width/height). */
  fovY?: number;
  aspect?: number;
  /** Polar angle range (from "up") the video covered, with margin. */
  polar?: [number, number];
}

export interface ShareLink {
  id: string;
  url: string;
  /** Epoch in ms; `null` when it never expires. */
  expiresAt: number | null;
  /** Only the link's creator has it: allows revoking before expiry. */
  deleteToken: string;
}

/** Lifetime in hours, or no expiry. */
export type ShareTtl = number | "never";

/** Dialog options for holders of the access key (the Worker accepts any
 *  value up to 10 years). Without the key, the server default applies. */
export const SHARE_TTLS: ShareTtl[] = [6, 24, 24 * 7, 24 * 30, 24 * 365, "never"];

/** Access key (the Worker's `SHARE_ADMIN_TOKEN`) and the requested lifetime. */
export interface ShareAccess {
  key: string;
  ttl: ShareTtl;
}

/** Worker error codes (translated in `share.errors`), plus client-side ones. */
export type ShareErrorCode =
  | "too_large"
  | "bad_format"
  | "rate_limited"
  | "forbidden"
  | "network"
  | "server"
  | (string & {});

export class ShareError extends Error {
  constructor(
    readonly code: ShareErrorCode,
    readonly maxBytes?: number,
  ) {
    super(`share: ${code}`);
  }
}

const sub = (a: number[], b: number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: number[]): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

const POLAR_MARGIN = (20 * Math.PI) / 180;

/** Polar angle range (from "up") covered by the video's cameras, with
 *  margin: outside it there is no data and the splat shows holes. */
export function polarRange(frame: SceneFrame): [number, number] {
  const up = norm(frame.up);
  let lo = Math.PI;
  let hi = 0;
  for (const c of frame.cameras) {
    const polar = Math.acos(Math.min(1, Math.max(-1, dot(norm(sub(c, frame.center)), up))));
    lo = Math.min(lo, polar);
    hi = Math.max(hi, polar);
  }
  return [Math.max(0, lo - POLAR_MARGIN), Math.min(Math.PI, hi + POLAR_MARGIN)];
}

/** Rounds so the link metadata fits comfortably within R2's limit. */
const r = (x: number) => Number(x.toPrecision(6));
const rv = (v: number[]): Vec3 => [r(v[0]), r(v[1]), r(v[2])];

export function shareView(frame: SceneFrame | null): ShareView | null {
  if (!frame?.cameras.length) return null;
  const [lo, hi] = polarRange(frame);
  const view: ShareView = {
    center: rv(frame.center),
    up: rv(norm(frame.up)),
    camera: rv(frame.cameras[0]),
    radius: r(frame.radius),
    polar: [r(lo), r(hi)],
  };
  if (frame.lens) {
    view.fovY = r(frame.lens.fovY);
    view.aspect = r(frame.lens.aspect);
  }
  return view;
}

/** Uploads the file and returns the link. XHR instead of fetch: only XHR
 *  reports upload progress. */
export function uploadShare(
  origin: string,
  file: Blob,
  name: string,
  view: ShareView | null,
  access: ShareAccess | null,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<ShareLink> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${origin}/api/share?name=${encodeURIComponent(name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    if (view) xhr.setRequestHeader("X-Share-View", JSON.stringify(view));
    if (access) {
      xhr.setRequestHeader("Authorization", `Bearer ${access.key}`);
      xhr.setRequestHeader("X-Share-TTL", String(access.ttl));
    }
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      const body = xhr.response as (ShareLink & { error?: string; maxBytes?: number }) | null;
      if (xhr.status === 201 && body?.url) resolve(body);
      else reject(new ShareError(body?.error ?? (xhr.status === 413 ? "too_large" : "server"), body?.maxBytes));
    };
    xhr.onerror = () => reject(new ShareError("network"));
    xhr.onabort = () => reject(new DOMException("cancelled", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(file);
  });
}

export async function revokeShare(origin: string, link: ShareLink) {
  const res = await fetch(`${origin}/api/share/${link.id}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${link.deleteToken}` },
  }).catch(() => null);
  if (!res) throw new ShareError("network");
  if (!res.ok) throw new ShareError("server");
}

// Links created on this device, per export: reopening the project shows the
// still-valid link instead of uploading again.
const STORE_KEY = "splatforge.shares.v1";

/** A redone export with the same name is a different file: include the size. */
export const shareKey = (path: string, bytes: number) => `${path}|${bytes}`;

function loadAll(): Record<string, ShareLink> {
  try {
    const all = JSON.parse(localStorage.getItem(STORE_KEY) ?? "{}") as Record<string, ShareLink>;
    const now = Date.now();
    return Object.fromEntries(Object.entries(all).filter(([, l]) => l.expiresAt === null || l.expiresAt > now));
  } catch {
    return {};
  }
}

export function savedShare(key: string): ShareLink | null {
  return loadAll()[key] ?? null;
}

export function saveShare(key: string, link: ShareLink | null) {
  const all = loadAll();
  if (link) all[key] = link;
  else delete all[key];
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(all));
  } catch {
    /* storage unavailable: the link only lasts this session */
  }
}

// Access key that unlocks choosing the lifetime; stays on this device only.
const ACCESS_KEY = "splatforge.shareAccess.v1";

/** `?admin=true` in the URL shows the key field and the lifetime picker.
 *  Without it, the panel is the regular one, even with a saved key. */
export const shareAdmin = (() => {
  try {
    return new URLSearchParams(location.search).get("admin") === "true";
  } catch {
    return false;
  }
})();

export function shareAccessKey(): string | null {
  try {
    return localStorage.getItem(ACCESS_KEY);
  } catch {
    return null;
  }
}

export function setShareAccessKey(key: string | null) {
  try {
    if (key) localStorage.setItem(ACCESS_KEY, key);
    else localStorage.removeItem(ACCESS_KEY);
  } catch {
    /* storage unavailable */
  }
}
