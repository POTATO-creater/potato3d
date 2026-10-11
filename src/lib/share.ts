// Compartilhamento por link: sobe a exportação para o Worker da versão web
// (worker/index.ts), que guarda no R2 por algumas horas e devolve um link
// privado para o visualizador (web/share.html). Desktop e web usam o mesmo
// código; o host só diz como ler o arquivo e onde fica o servidor.

import type { SceneFrame } from "./api";

type Vec3 = [number, number, number];

/** Enquadramento do visualizador do link, nas coordenadas do splat: o mesmo
 *  "up", centro e ponto de vista inicial do visualizador do app. */
export interface ShareView {
  center: Vec3;
  up: Vec3;
  /** Câmera do primeiro frame do vídeo. */
  camera: Vec3;
  /** Distância média das câmeras ao centro. */
  radius: number;
  /** Lente do vídeo (campo de visão vertical em radianos e largura/altura). */
  fovY?: number;
  aspect?: number;
  /** Faixa de ângulo polar (a partir do "up") que o vídeo filmou, com folga. */
  polar?: [number, number];
}

export interface ShareLink {
  id: string;
  url: string;
  /** Epoch em ms. */
  expiresAt: number;
  /** Só quem criou o link tem: permite revogar antes de expirar. */
  deleteToken: string;
}

/** Códigos de erro do Worker (traduzidos em `share.errors`), mais os do cliente. */
export type ShareErrorCode =
  | "too_large"
  | "bad_format"
  | "rate_limited"
  | "network"
  | "server"
  | (string & {});

export class ShareError extends Error {
  constructor(
    readonly code: ShareErrorCode,
    readonly maxBytes?: number,
  ) {
    super(`compartilhamento: ${code}`);
  }
}

const sub = (a: number[], b: number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: number[]): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

const POLAR_MARGIN = (20 * Math.PI) / 180;

/** Faixa de ângulo polar (a partir do "up") coberta pelas câmeras do vídeo,
 *  com folga: fora dela não há dados e o splat mostra buracos. */
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

/** Arredonda para os metadados do link caberem com folga no limite do R2. */
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

/** Sobe o arquivo e devolve o link. XHR em vez de fetch: só ele dá o
 *  progresso do envio. */
export function uploadShare(
  origin: string,
  file: Blob,
  name: string,
  view: ShareView | null,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<ShareLink> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${origin}/api/share?name=${encodeURIComponent(name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    if (view) xhr.setRequestHeader("X-Share-View", JSON.stringify(view));
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      const body = xhr.response as (ShareLink & { error?: string; maxBytes?: number }) | null;
      if (xhr.status === 201 && body?.url) resolve(body);
      else reject(new ShareError(body?.error ?? (xhr.status === 413 ? "too_large" : "server"), body?.maxBytes));
    };
    xhr.onerror = () => reject(new ShareError("network"));
    xhr.onabort = () => reject(new DOMException("cancelado", "AbortError"));
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

// Links criados neste aparelho, por exportação: reabrir o projeto mostra o
// link ainda válido em vez de subir de novo.
const STORE_KEY = "splatforge.shares.v1";

/** Uma exportação refeita com o mesmo nome é outro arquivo: entra o tamanho. */
export const shareKey = (path: string, bytes: number) => `${path}|${bytes}`;

function loadAll(): Record<string, ShareLink> {
  try {
    const all = JSON.parse(localStorage.getItem(STORE_KEY) ?? "{}") as Record<string, ShareLink>;
    const now = Date.now();
    return Object.fromEntries(Object.entries(all).filter(([, l]) => l.expiresAt > now));
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
    /* armazenamento indisponível: o link só vale nesta sessão */
  }
}
