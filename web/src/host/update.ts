// Service worker (offline cache, web/src/sw.template.js) and new-version
// notice: once a new build finishes downloading and is waiting, the UI shows
// "Update" (Host.onUpdate / Host.applyUpdate).

/** New version installed and waiting to take over. */
let waiting: ServiceWorker | null = null;
const listeners = new Set<() => void>();

// Check for new deploys periodically (training sessions are long).
const CHECK_EVERY_MS = 60 * 60 * 1000;

function announce(w: ServiceWorker) {
  waiting = w;
  for (const h of listeners) h();
}

/** Build only: sw.js doesn't exist in dev. Registers after load so the ~60 MB
 *  cache download doesn't compete with the app's startup. */
export function registerServiceWorker() {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) return;
  window.addEventListener("load", async () => {
    try {
      const reg = await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`);
      // No controller means first install: there is no "old" version to replace.
      const track = (w: ServiceWorker) =>
        w.addEventListener("statechange", () => {
          if (w.state === "installed" && navigator.serviceWorker.controller) announce(w);
        });
      if (reg.waiting && navigator.serviceWorker.controller) announce(reg.waiting);
      if (reg.installing) track(reg.installing);
      reg.addEventListener("updatefound", () => reg.installing && track(reg.installing));
      setInterval(() => reg.update().catch(() => {}), CHECK_EVERY_MS);
    } catch (e) {
      console.warn("service worker:", e);
    }
  });
}

export function onUpdate(handler: () => void): () => void {
  listeners.add(handler);
  if (waiting) handler();
  return () => void listeners.delete(handler);
}

/** Activates the new version and reloads this tab. If another tab already did
 *  that, just reload. */
export function applyUpdate() {
  if (!waiting || waiting.state !== "installed") {
    location.reload();
    return;
  }
  navigator.serviceWorker.addEventListener("controllerchange", () => location.reload(), { once: true });
  waiting.postMessage({ type: "skipWaiting" });
}
