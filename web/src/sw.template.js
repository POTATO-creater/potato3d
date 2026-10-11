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
// O app é guardado na raiz do escopo, não em index.html: o Cloudflare responde
// /index.html com 307 para /, e o Chrome recusa resposta redirecionada numa
// navegação (ERR_FAILED ao recarregar).
const INDEX = scoped("./");
const PRECACHE = new Set(SW_BUILD.files.map((f) => (f === "index.html" ? INDEX : scoped(f))));

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

/** Navegação não aceita resposta com `redirected` (de um cache antigo ou de
 *  um servidor que redirecione a raiz): entrega uma cópia limpa. */
async function unredirect(res) {
  if (!res.redirected) return res;
  return new Response(await res.blob(), { status: res.status, statusText: res.statusText, headers: res.headers });
}

/** Um arquivo de outra versão que já saiu do ar: o Cloudflare responde com o
 *  index.html (fallback de SPA), e um worker carregado assim falha sem
 *  mensagem. Um 404 deixa o erro claro. */
function notIndex(res) {
  const html = res.headers.get("Content-Type")?.startsWith("text/html");
  return html ? new Response("not found", { status: 404 }) : res;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  // Links compartilhados e a API deles são do Worker (worker/index.ts):
  // nunca do cache, e uma navegação para /s/<id> não pode abrir o app.
  if (url.pathname.startsWith("/s/") || url.pathname.startsWith("/api/")) return;

  // Navegações (inclusive com ?query, como ?dataset=…) abrem o app do cache.
  if (request.mode === "navigate" && url.href.startsWith(self.registration.scope)) {
    event.respondWith(fromCache(INDEX, request).then(unredirect));
    return;
  }
  const path = url.origin + url.pathname;
  const clean = path === scoped("index.html") ? INDEX : path;
  // Arquivos com hash de outra versão: uma aba antiga pedindo os dela.
  const old = clean.startsWith(scoped("assets/"));
  if (PRECACHE.has(clean)) event.respondWith(fromCache(clean, request));
  else if (old) event.respondWith(fromCache(clean, request).then(notIndex));
  // O resto (vídeo e dataset de teste via ?video=/?dataset=) vai direto à rede.
});
