// Web service worker: makes the whole app available offline.
//
// The build (web/sw-plugin.ts) generates sw.js by prefixing this file with
// `const SW_BUILD = { version, files }`: the version is a hash of the build's
// content and `files` lists everything it produced (~60 MB, mostly the Brush
// and ONNX Runtime WASM), relative to the scope.
//
// Lifecycle: each version has its own cache, fully downloaded at install time.
// A new version doesn't take over mid-session on its own: a tab open on the old
// version will still lazily load workers and WASM (frames, poses, training),
// and those must come from the same build. It takes over when every tab closes
// or when the user clicks "Update" (`skipWaiting` message, see
// web/src/host/update.ts). In the latter case, if other tabs are open, the
// previous version's cache is kept for them (fetch searches every cache) and
// is only removed on the following update.
//
// The RMBG model doesn't go through here: the app stores it in OPFS itself.

/* global SW_BUILD */
const PREFIX = "gausify-orbit-";
const CACHE = PREFIX + SW_BUILD.version;
/** Stores the active version, to know which one is "the previous" on the next activation. */
const META = PREFIX + "meta";
const ACTIVE_KEY = "/__gausify-orbit-active-version__";
const scoped = (path) => new URL(path, self.registration.scope).href;
// The app is cached at the scope root, not at index.html: Cloudflare answers
// /index.html with a 307 to /, and Chrome rejects a redirected response for a
// navigation (ERR_FAILED on reload).
const INDEX = scoped("./");
const PRECACHE = new Set(SW_BUILD.files.map((f) => (f === "index.html" ? INDEX : scoped(f))));

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      // `reload`: bypass the HTTP cache (index.html and the icons have no hash in their name).
      await cache.addAll([...PRECACHE].map((url) => new Request(url, { cache: "reload" })));
    })(),
  );
});

/** Set by a tab's "Update" button, instead of waiting for every tab to close. */
let keepPrevious = false;
self.addEventListener("message", (event) => {
  if (event.data?.type !== "skipWaiting") return;
  event.waitUntil(
    (async () => {
      // includeUncontrolled: the open tabs belong to the previous version, not this one.
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
      // On first install, also start serving the tab that registered it.
      await self.clients.claim();
    })(),
  );
});

/** From this version's cache; else from a previous one (tab opened before the
 *  update); else from the network. */
async function fromCache(url, request) {
  const hit = (await (await caches.open(CACHE)).match(url)) ?? (await caches.match(url));
  return hit ?? fetch(request);
}

/** Navigations reject a `redirected` response (from an old cache or a server
 *  that redirects the root): hand back a clean copy. */
async function unredirect(res) {
  if (!res.redirected) return res;
  return new Response(await res.blob(), { status: res.status, statusText: res.statusText, headers: res.headers });
}

/** A file from another version that is no longer deployed: Cloudflare answers
 *  with index.html (SPA fallback), and a worker loaded that way fails with no
 *  message. A 404 makes the error clear. */
function notIndex(res) {
  const html = res.headers.get("Content-Type")?.startsWith("text/html");
  return html ? new Response("not found", { status: 404 }) : res;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  // Share links and their API belong to the Worker (worker/index.ts): never
  // from the cache, and a navigation to /s/<id> must not open the app.
  if (url.pathname.startsWith("/s/") || url.pathname.startsWith("/api/")) return;

  // Navigations (including with a ?query, like ?dataset=…) open the app from the cache.
  if (request.mode === "navigate" && url.href.startsWith(self.registration.scope)) {
    event.respondWith(fromCache(INDEX, request).then(unredirect));
    return;
  }
  const path = url.origin + url.pathname;
  const clean = path === scoped("index.html") ? INDEX : path;
  // Hashed files from another version: an old tab requesting its own.
  const old = clean.startsWith(scoped("assets/"));
  if (PRECACHE.has(clean)) event.respondWith(fromCache(clean, request));
  else if (old) event.respondWith(fromCache(clean, request).then(notIndex));
  // Everything else (test video and dataset via ?video=/?dataset=) goes straight to the network.
});
