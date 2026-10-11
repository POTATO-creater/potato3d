// Worker da versão web. Os assets do build (dist-web/) continuam servidos
// direto pelo Cloudflare, sem acordar o Worker; ele só roda em:
//
//   POST   /api/share            sobe um .sog/.ply exportado e devolve o link
//   GET    /api/share/<id>       metadados (nome, tamanho, expiração, enquadramento)
//   GET    /api/share/<id>/file  o arquivo
//   DELETE /api/share/<id>       revoga antes da hora (precisa do deleteToken)
//   GET    /s/<id>               página do visualizador (web/share.html)
//
// Os compartilhamentos ficam no R2 (`SHARES`) com a expiração nos metadados.
// Quem tem o link vê; o id tem ~71 bits aleatórios, então não dá para
// adivinhar. Expirado é tratado como inexistente na hora; a limpeza de fato
// é o cron (`scheduled`), com a regra de ciclo de vida do bucket como rede
// de segurança (ver README).
//
// Com a chave de acesso (`SHARE_ADMIN_TOKEN`), o envio escolhe a validade em
// `X-Share-TTL` (horas, ou `never`). Esses ficam em `p/`, fora da regra de
// ciclo de vida de `s/`, que apagaria tudo em um dia; o cron continua
// cuidando deles.

export interface Env {
  ASSETS: { fetch(input: Request | string): Promise<Response> };
  SHARES: R2Bucket;
  /** Validade de cada link, em horas (padrão 6). */
  SHARE_TTL_HOURS?: string;
  /** Segredo (`wrangler secret put`) que libera validade escolhida no envio,
   *  inclusive sem expiração. Ausente, todo link vale `SHARE_TTL_HOURS`. */
  SHARE_ADMIN_TOKEN?: string;
  /** Tamanho máximo do arquivo, em MiB (padrão 95; o Cloudflare recusa
   *  corpos acima de 100 MB nos planos Free e Pro). */
  SHARE_MAX_MB?: string;
  /** Origens extras com CORS liberado, separadas por vírgula (o app desktop
   *  já está incluído). */
  SHARE_ORIGINS?: string;
  /** Limite de uploads por IP (binding `ratelimits`); ausente, não limita. */
  SHARE_LIMITER?: { limit(o: { key: string }): Promise<{ success: boolean }> };
}

/** O que o app manda para o visualizador abrir no mesmo enquadramento.
 *  Espelha `ShareView` de src/lib/share.ts. */
interface ShareView {
  center: number[];
  up: number[];
  camera: number[];
  radius: number;
  fovY?: number;
  aspect?: number;
  polar?: number[];
}

const ID_LEN = 12;
const ID_RE = /^[0-9A-Za-z]{12}$/;
const FORMATS = { sog: "PK\x03\x04", ply: "ply\n" } as const;
type Format = keyof typeof FORMATS;
// Webview do app desktop no macOS/Linux e no Windows.
const DESKTOP_ORIGINS = ["tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"];

/** `s/`: validade padrão, coberta pela regra de ciclo de vida do bucket.
 *  `p/`: validade escolhida com a chave de acesso. */
const PREFIXES = ["s/", "p/"] as const;
/** Nos metadados, no lugar do epoch, para o link que não expira. */
const NEVER = "never";
/** Teto do cache privado do arquivo: um link longo ainda pode ser revogado. */
const MAX_CACHE_S = 86_400;
/** Maior validade com a chave, fora `never` (10 anos). */
const MAX_TTL_HOURS = 24 * 3653;

/** Acha o compartilhamento em qualquer prefixo (`s/` primeiro: é o comum). */
async function findObject(env: Env, id: string, body: boolean) {
  for (const prefix of PREFIXES) {
    const key = prefix + id;
    const obj = body ? await env.SHARES.get(key) : await env.SHARES.head(key);
    if (obj) return obj;
  }
  return null;
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    const path = url.pathname;
    if (path === "/api/share" || path.startsWith("/api/share/")) {
      return cors(req, env, await api(req, env, ctx, url).catch(fail));
    }
    if (path.startsWith("/s/")) return viewerPage(env, url);
    return env.ASSETS.fetch(req);
  },

  async scheduled(_event: ScheduledController, env: Env, ctx: ExecutionContext) {
    ctx.waitUntil(sweep(env));
  },
};

async function api(req: Request, env: Env, ctx: ExecutionContext, url: URL): Promise<Response> {
  if (req.method === "OPTIONS") return new Response(null, { status: 204 });
  const [, , , id, sub] = url.pathname.split("/");
  if (id === undefined) {
    return req.method === "POST" ? create(req, env, url) : error(405, "method");
  }
  if (!ID_RE.test(id) || (sub !== undefined && sub !== "file")) return error(404, "not_found");
  if (req.method === "DELETE" && sub === undefined) return revoke(req, env, id);
  if (req.method !== "GET" && req.method !== "HEAD") return error(405, "method");

  const obj = await findObject(env, id, sub === "file");
  const meta = obj && readMeta(obj);
  if (!obj || !meta || (meta.expiresAt !== null && meta.expiresAt <= Date.now())) {
    if (obj) ctx.waitUntil(env.SHARES.delete(obj.key));
    return error(404, "not_found");
  }
  const left = meta.expiresAt === null ? MAX_CACHE_S : Math.floor((meta.expiresAt - Date.now()) / 1000);
  const maxAge = Math.max(0, Math.min(left, MAX_CACHE_S));
  if (sub === undefined) {
    return json(200, {
      name: meta.name,
      format: meta.format,
      size: obj.size,
      expiresAt: meta.expiresAt,
      view: meta.view,
    });
  }
  return new Response((obj as R2ObjectBody).body, {
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Length": String(obj.size),
      "Content-Disposition": `inline; filename="${meta.name}"`,
      // Privado: não fica em cache compartilhado, e some junto com o link.
      "Cache-Control": `private, max-age=${maxAge}`,
      ETag: obj.httpEtag,
      "X-Robots-Tag": "noindex, nofollow",
    },
  });
}

async function create(req: Request, env: Env, url: URL): Promise<Response> {
  if (env.SHARE_LIMITER) {
    const ip = req.headers.get("CF-Connecting-IP") ?? "local";
    if (!(await env.SHARE_LIMITER.limit({ key: ip })).success) return error(429, "rate_limited");
  }
  const name = safeName(url.searchParams.get("name") ?? "");
  const format = name.split(".").pop() as Format;
  if (!(format in FORMATS)) return error(400, "bad_format");

  const size = Number(req.headers.get("Content-Length"));
  if (!req.body || !Number.isFinite(size) || size <= 0) return error(411, "no_length");
  const maxBytes = number(env.SHARE_MAX_MB, 95) * 1024 * 1024;
  if (size > maxBytes) return error(413, "too_large", { maxBytes });

  let view: string | null = null;
  const rawView = req.headers.get("X-Share-View");
  if (rawView) {
    view = parseView(rawView);
    if (!view) return error(400, "bad_view");
  }

  // Validade escolhida: só com a chave de acesso.
  let ttl: number | typeof NEVER = number(env.SHARE_TTL_HOURS, 6);
  const rawTtl = req.headers.get("X-Share-TTL");
  if (rawTtl !== null) {
    if (!(await isAdmin(req, env))) return error(403, "forbidden");
    const hours = rawTtl === NEVER ? NEVER : Number(rawTtl);
    if (hours !== NEVER && !(Number.isFinite(hours) && hours > 0 && hours <= MAX_TTL_HOURS)) {
      return error(400, "bad_ttl");
    }
    ttl = hours;
  }

  const id = randomId(ID_LEN);
  const deleteToken = randomId(32);
  const expiresAt = ttl === NEVER ? null : Date.now() + ttl * 3600_000;
  const key = (rawTtl === null ? "s/" : "p/") + id;
  await env.SHARES.put(key, req.body, {
    httpMetadata: { contentType: "application/octet-stream" },
    customMetadata: {
      name,
      format,
      expiresAt: expiresAt === null ? NEVER : String(expiresAt),
      del: await sha256(deleteToken),
      ...(view ? { view } : {}),
    },
  });

  // Confere a assinatura do arquivo depois de gravado (o corpo é um stream e
  // só passa uma vez): o endpoint não serve para hospedar outra coisa.
  const head = await env.SHARES.get(key, { range: { offset: 0, length: 4 } });
  const magic = head ? new TextDecoder("latin1").decode(await head.arrayBuffer()) : "";
  if (magic !== FORMATS[format]) {
    await env.SHARES.delete(key);
    return error(400, "bad_format");
  }
  return json(201, { id, url: `${url.origin}/s/${id}`, expiresAt, deleteToken });
}

async function revoke(req: Request, env: Env, id: string): Promise<Response> {
  const token = req.headers.get("Authorization")?.replace(/^Bearer\s+/i, "") ?? "";
  const obj = await findObject(env, id, false);
  // Já expirado ou removido: para quem revoga, dá no mesmo.
  if (!obj) return new Response(null, { status: 204 });
  if (!token || (await sha256(token)) !== obj.customMetadata?.del) return error(403, "forbidden");
  await env.SHARES.delete(obj.key);
  return new Response(null, { status: 204 });
}

/** A página do visualizador é a mesma para todo id: ela própria busca os
 *  metadados e mostra "expirado" quando for o caso. */
async function viewerPage(env: Env, url: URL): Promise<Response> {
  // `/share` e não `/share.html`: o Cloudflare redireciona o .html.
  const page = await env.ASSETS.fetch(new Request(new URL("/share", url)));
  const headers = new Headers(page.headers);
  headers.set("Cache-Control", "no-store");
  headers.set("X-Robots-Tag", "noindex, nofollow");
  // O link é a credencial: não pode vazar no Referer de nada que a página carregue.
  headers.set("Referrer-Policy", "no-referrer");
  return new Response(page.body, { status: page.status, headers });
}

/** Apaga os vencidos. Roda pelo cron do wrangler.jsonc. */
export async function sweep(env: Env, now = Date.now()) {
  let removed = 0;
  for (const prefix of PREFIXES) {
    let cursor: string | undefined;
    do {
      const page = await env.SHARES.list({ prefix, cursor, include: ["customMetadata"] });
      const expired = page.objects
        .filter((o) => o.customMetadata?.expiresAt !== NEVER && !(Number(o.customMetadata?.expiresAt) > now))
        .map((o) => o.key);
      if (expired.length) await env.SHARES.delete(expired);
      removed += expired.length;
      cursor = page.truncated ? page.cursor : undefined;
    } while (cursor);
  }
  return removed;
}

function readMeta(obj: R2Object) {
  const m = obj.customMetadata ?? {};
  const expiresAt = m.expiresAt === NEVER ? null : Number(m.expiresAt);
  if ((expiresAt !== null && !Number.isFinite(expiresAt)) || !(m.format in FORMATS)) return null;
  return {
    name: m.name || `splat.${m.format}`,
    format: m.format as Format,
    expiresAt,
    view: m.view ? (JSON.parse(m.view) as ShareView) : null,
  };
}

/** Valida e reescreve o enquadramento (só números finitos, tamanho fixo). */
function parseView(raw: string): string | null {
  try {
    const v = JSON.parse(raw) as Partial<ShareView>;
    const vec = (a: unknown, n: number) =>
      Array.isArray(a) && a.length === n && a.every((x) => typeof x === "number" && Number.isFinite(x));
    const num = (x: unknown) => typeof x === "number" && Number.isFinite(x);
    if (!vec(v.center, 3) || !vec(v.up, 3) || !vec(v.camera, 3) || !num(v.radius)) return null;
    if (v.fovY !== undefined && !num(v.fovY)) return null;
    if (v.aspect !== undefined && !num(v.aspect)) return null;
    if (v.polar !== undefined && !vec(v.polar, 2)) return null;
    const clean: ShareView = { center: v.center!, up: v.up!, camera: v.camera!, radius: v.radius! };
    if (v.fovY !== undefined) clean.fovY = v.fovY;
    if (v.aspect !== undefined) clean.aspect = v.aspect;
    if (v.polar !== undefined) clean.polar = v.polar;
    return JSON.stringify(clean);
  } catch {
    return null;
  }
}

/** Nome de arquivo seguro para metadados e Content-Disposition. */
function safeName(name: string) {
  const base = name.split(/[\\/]/).pop() ?? "";
  return base.replace(/[^\w.-]+/g, "_").slice(-80) || "splat";
}

function cors(req: Request, env: Env, res: Response): Response {
  const origin = req.headers.get("Origin");
  if (!origin) return res;
  const allowed = [...DESKTOP_ORIGINS, ...(env.SHARE_ORIGINS ?? "").split(",").map((s) => s.trim())];
  if (!allowed.includes(origin)) return res;
  const out = new Response(res.body, res);
  out.headers.set("Access-Control-Allow-Origin", origin);
  out.headers.set("Access-Control-Allow-Methods", "GET, POST, DELETE");
  out.headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Share-View, X-Share-TTL");
  out.headers.set("Access-Control-Max-Age", "86400");
  out.headers.append("Vary", "Origin");
  return out;
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store", "X-Robots-Tag": "noindex" },
  });
}

/** Erros com um código estável, que o app traduz (src/i18n, `share.errors`). */
const error = (status: number, code: string, extra?: object) => json(status, { error: code, ...extra });

function fail(e: unknown): Response {
  console.error(e);
  return error(500, "server");
}

function number(raw: string | undefined, fallback: number) {
  const n = Number(raw);
  return raw && Number.isFinite(n) && n > 0 ? n : fallback;
}

const ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/** Base62 sem viés: descarta os bytes acima do maior múltiplo de 62. */
export function randomId(len: number) {
  let out = "";
  while (out.length < len) {
    for (const b of crypto.getRandomValues(new Uint8Array(len * 2))) {
      if (b < 248 && out.length < len) out += ALPHABET[b % 62];
    }
  }
  return out;
}

/** Compara os hashes, não a chave: o tempo não diz quantos caracteres batem. */
async function isAdmin(req: Request, env: Env) {
  const token = req.headers.get("Authorization")?.replace(/^Bearer\s+/i, "") ?? "";
  return !!env.SHARE_ADMIN_TOKEN && !!token && (await sha256(token)) === (await sha256(env.SHARE_ADMIN_TOKEN));
}

async function sha256(text: string) {
  const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
