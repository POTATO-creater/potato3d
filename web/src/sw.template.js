// Service worker da versão web: deixa o app inteiro disponível offline.
//
// O build (web/sw-plugin.ts) gera o sw.js prefixando este arquivo com
// `const SW_BUILD = { version, files }`: a versão é um hash do conteúdo do
// build e `files` lista tudo que ele gerou (~60 MB, quase tudo WASM do Brush
// e do ONNX Runtime), relativo ao escopo.
//
// Ciclo de vida: cada versão tem o próprio cache, baixado inteiro na
// instalação. A versão nova não assume sozinha no meio do uso: uma aba aberta
// com a versão antiga ainda vai carregar workers e WASM sob demanda (frames,
// poses, treino), e eles têm que vir do mesmo build. Ela entra quando todas as
// abas fecham ou quando o usuário clica em "Atualizar" (mensagem
// `skipWaiting`, ver web/src/host/update.ts). Nesse segundo caso, se houver
// outras abas abertas, o cache da versão anterior fica guardado para elas
// (o fetch procura em todos os caches) e só some na atualização seguinte.
//
// O modelo RMBG não passa por aqui: ele é guardado no OPFS pelo próprio app.

/* global SW_BUILD */
const PREFIX = "gausify-orbit-";
const CACHE = PREFIX + SW_BUILD.version;
/** Guarda a versão ativa, para saber qual é "a anterior" na próxima ativação. */
const META = PREFIX + "meta";
const ACTIVE_KEY = "/__gausify-orbit-active-version__";
const scoped = (path) => new URL(path, self.registration.scope).href;
const PRECACHE = new Set(SW_BUILD.files.map(scoped));
const INDEX = scoped("index.html");

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      // `reload`: não aproveita o cache HTTP (index.html e ícones não têm hash no nome).
      await cache.addAll([...PRECACHE].map((url) => new Request(url, { cache: "reload" })));
    })(),
  );
});

/** Ativada pelo botão "Atualizar" de uma aba, em vez de esperar todas fecharem. */
let keepPrevious = false;
self.addEventListener("message", (event) => {
  if (event.data?.type !== "skipWaiting") return;
  event.waitUntil(
    (async () => {
      // includeUncontrolled: as abas abertas são da versão anterior, não desta.
      keepPrevious = (await self.clients.matchAll({ type: "window", includeUncontrolled: true })).length > 1;
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const meta = await caches.open(META);
      const previous = await (await meta.match(ACTIVE_KEY))?.text();
      const keep = new Set([CACHE, META]);
      if (keepPrevious && previous) keep.add(PREFIX + previous);
      for (const name of await caches.keys()) {
        if (name.startsWith(PREFIX) && !keep.has(name)) await caches.delete(name);
      }
      await meta.put(ACTIVE_KEY, new Response(SW_BUILD.version));
      // Na primeira instalação, passa a servir também a aba que a registrou.
      await self.clients.claim();
    })(),
  );
});

/** Do cache desta versão; senão de uma anterior (aba aberta antes de
 *  atualizar); senão da rede. */
async function fromCache(url, request) {
  const hit = (await (await caches.open(CACHE)).match(url)) ?? (await caches.match(url));
  return hit ?? fetch(request);
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // Navegações (inclusive com ?query, como ?dataset=…) abrem o app do cache.
  if (request.mode === "navigate" && url.href.startsWith(self.registration.scope)) {
    event.respondWith(fromCache(INDEX, request));
    return;
  }
  const clean = url.origin + url.pathname;
  // Arquivos com hash de outra versão: uma aba antiga pedindo os dela.
  const old = clean.startsWith(scoped("assets/"));
  if (PRECACHE.has(clean) || old) event.respondWith(fromCache(clean, request));
  // O resto (vídeo e dataset de teste via ?video=/?dataset=) vai direto à rede.
});
