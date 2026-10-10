// Service worker (cache offline, web/src/sw.template.js) e aviso de versão
// nova: quando um build novo termina de baixar e fica esperando, a interface
// mostra "Atualizar" (Host.onUpdate / Host.applyUpdate).

/** Versão nova instalada e esperando para assumir. */
let waiting: ServiceWorker | null = null;
const listeners = new Set<() => void>();

// Procura deploys novos de tempos em tempos (sessões de treino são longas).
const CHECK_EVERY_MS = 60 * 60 * 1000;

function announce(w: ServiceWorker) {
  waiting = w;
  for (const h of listeners) h();
}

/** Só no build: no dev o sw.js não existe. Registra depois do load para os
 *  ~60 MB do cache não disputarem a abertura do app. */
export function registerServiceWorker() {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) return;
  window.addEventListener("load", async () => {
    try {
      const reg = await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`);
      // Sem controller é a primeira instalação: não há versão "antiga" a trocar.
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

/** Põe a versão nova no ar e recarrega esta aba. Se outra aba já fez isso,
 *  basta recarregar. */
export function applyUpdate() {
  if (!waiting || waiting.state !== "installed") {
    location.reload();
    return;
  }
  navigator.serviceWorker.addEventListener("controllerchange", () => location.reload(), { once: true });
  waiting.postMessage({ type: "skipWaiting" });
}
