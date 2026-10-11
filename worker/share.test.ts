// Testes do Worker de compartilhamento com um R2 em memória.
// `pnpm test:web` (node --test, que lê TypeScript direto).

import { test } from "node:test";
import assert from "node:assert/strict";
import worker, { sweep, type Env } from "./index.ts";

type Stored = { bytes: Uint8Array; customMetadata: Record<string, string> };

/** O pedaço da API do R2 que o Worker usa. */
function fakeBucket() {
  const objects = new Map<string, Stored>();
  const object = (key: string, o: Stored, bytes?: Uint8Array) => ({
    key,
    size: o.bytes.length,
    httpEtag: `"${key}"`,
    customMetadata: o.customMetadata,
    ...(bytes && {
      body: new Response(bytes).body,
      arrayBuffer: async () => bytes.slice().buffer,
    }),
  });
  const bucket = {
    objects,
    async put(key: string, body: ReadableStream, opts: { customMetadata: Record<string, string> }) {
      objects.set(key, { bytes: new Uint8Array(await new Response(body).arrayBuffer()), customMetadata: opts.customMetadata });
    },
    async head(key: string) {
      const o = objects.get(key);
      return o ? object(key, o) : null;
    },
    async get(key: string, opts?: { range?: { offset: number; length: number } }) {
      const o = objects.get(key);
      if (!o) return null;
      const r = opts?.range;
      return object(key, o, r ? o.bytes.slice(r.offset, r.offset + r.length) : o.bytes);
    },
    async delete(keys: string | string[]) {
      for (const k of [keys].flat()) objects.delete(k);
    },
    async list({ prefix }: { prefix: string }) {
      const keys = [...objects.keys()].filter((k) => k.startsWith(prefix));
      return { objects: keys.map((k) => object(k, objects.get(k)!)), truncated: false };
    },
  };
  return bucket;
}

function setup(vars: Partial<Env> = {}) {
  const bucket = fakeBucket();
  const assetsHit: string[] = [];
  const env = {
    SHARES: bucket,
    ASSETS: {
      async fetch(input: Request | string) {
        const url = new URL(typeof input === "string" ? input : input.url);
        assetsHit.push(url.pathname);
        return new Response(`asset ${url.pathname}`, { headers: { "Content-Type": "text/html" } });
      },
    },
    ...vars,
  } as unknown as Env;
  const waits: Promise<unknown>[] = [];
  const ctx = { waitUntil: (p: Promise<unknown>) => waits.push(p), passThroughOnException() {} } as unknown as ExecutionContext;
  const call = (path: string, init?: RequestInit) =>
    worker.fetch(new Request(`https://orbit.test${path}`, init), env, ctx).then(async (r) => {
      await Promise.all(waits);
      return r;
    });
  return { bucket, env, call, assetsHit };
}

const SOG = new Uint8Array([0x50, 0x4b, 3, 4, 1, 2, 3, 4, 5]);
const VIEW = { center: [0, 1, 2], up: [0, -1, 0], camera: [3, 1, 2], radius: 3, fovY: 0.6, aspect: 1.77, polar: [0.5, 1.9] };

function upload(call: ReturnType<typeof setup>["call"], bytes: Uint8Array = SOG, name = "splat.sog", headers: Record<string, string> = {}) {
  return call(`/api/share?name=${encodeURIComponent(name)}`, {
    method: "POST",
    body: bytes,
    headers: { "Content-Length": String(bytes.length), "X-Share-View": JSON.stringify(VIEW), ...headers },
  });
}

test("sobe, lê os metadados e baixa o arquivo", async () => {
  const { call } = setup();
  const res = await upload(call);
  assert.equal(res.status, 201);
  const link = (await res.json()) as { id: string; url: string; expiresAt: number; deleteToken: string };
  assert.match(link.id, /^[0-9A-Za-z]{12}$/);
  assert.equal(link.url, `https://orbit.test/s/${link.id}`);
  // Validade padrão: 6 h.
  assert.ok(Math.abs(link.expiresAt - (Date.now() + 6 * 3600_000)) < 5000);

  const meta = (await (await call(`/api/share/${link.id}`)).json()) as Record<string, unknown>;
  assert.deepEqual(meta, { name: "splat.sog", format: "sog", size: SOG.length, expiresAt: link.expiresAt, view: VIEW });

  const file = await call(`/api/share/${link.id}/file`);
  assert.equal(file.status, 200);
  assert.deepEqual(new Uint8Array(await file.arrayBuffer()), SOG);
  assert.match(file.headers.get("Cache-Control")!, /^private, max-age=\d+$/);
});

test("validade e tamanho máximo vêm das variáveis", async () => {
  const { call } = setup({ SHARE_TTL_HOURS: "0.5", SHARE_MAX_MB: "0.000006" });
  assert.equal((await upload(call)).status, 413);
  const small = new Uint8Array([0x70, 0x6c, 0x79, 0x0a]); // "ply\n"
  const res = await upload(call, small, "a.ply");
  assert.equal(res.status, 201);
  const { expiresAt } = (await res.json()) as { expiresAt: number };
  assert.ok(Math.abs(expiresAt - (Date.now() + 1800_000)) < 5000);
});

test("recusa o que não é .sog/.ply de verdade", async () => {
  const { call, bucket } = setup();
  assert.equal((await upload(call, SOG, "x.zip")).status, 400);
  assert.equal((await upload(call, new TextEncoder().encode("<html>"), "x.sog")).status, 400);
  assert.equal((await upload(call, SOG, "x.sog", { "X-Share-View": '{"center":[1]}' })).status, 400);
  assert.equal(bucket.objects.size, 0);
});

test("expirado some na hora e no cron", async () => {
  const { call, bucket, env } = setup();
  const { id } = (await (await upload(call)).json()) as { id: string };
  const { id: other } = (await (await upload(call)).json()) as { id: string };
  bucket.objects.get(`s/${id}`)!.customMetadata.expiresAt = String(Date.now() - 1);
  assert.equal((await call(`/api/share/${id}/file`)).status, 404);
  assert.equal(bucket.objects.has(`s/${id}`), false);

  bucket.objects.get(`s/${other}`)!.customMetadata.expiresAt = String(Date.now() - 1);
  assert.equal(await sweep(env), 1);
  assert.equal(bucket.objects.size, 0);
});

test("revogar exige o token", async () => {
  const { call } = setup();
  const { id, deleteToken } = (await (await upload(call)).json()) as { id: string; deleteToken: string };
  const del = (token: string) => call(`/api/share/${id}`, { method: "DELETE", headers: { Authorization: `Bearer ${token}` } });
  assert.equal((await del("errado")).status, 403);
  assert.equal((await del(deleteToken)).status, 204);
  assert.equal((await call(`/api/share/${id}`)).status, 404);
  assert.equal((await del(deleteToken)).status, 204);
});

test("CORS só para o app desktop e origens configuradas", async () => {
  const { call } = setup({ SHARE_ORIGINS: "http://localhost:1420" });
  const pre = (origin: string) => call("/api/share", { method: "OPTIONS", headers: { Origin: origin } });
  assert.equal((await pre("tauri://localhost")).headers.get("Access-Control-Allow-Origin"), "tauri://localhost");
  assert.equal((await pre("http://localhost:1420")).headers.get("Access-Control-Allow-Origin"), "http://localhost:1420");
  assert.equal((await pre("https://evil.example")).headers.get("Access-Control-Allow-Origin"), null);
});

test("rotas fora da API", async () => {
  const { call, assetsHit } = setup();
  const page = await call("/s/abcdefABCDEF");
  assert.equal(await page.text(), "asset /share");
  assert.equal(page.headers.get("Referrer-Policy"), "no-referrer");
  await call("/");
  assert.deepEqual(assetsHit, ["/share", "/"]);
});
