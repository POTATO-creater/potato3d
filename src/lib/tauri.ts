// Host do app desktop: comandos do backend em Rust via Tauri.

import { Channel, convertFileSrc, invoke } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import type { Api, ExportStats, Host, InstallEvent, InstallPlan, JobEvent } from "./api";

function channel<T>(onMessage: (m: T) => void) {
  const c = new Channel<T>();
  c.onmessage = onMessage;
  return c;
}

const api: Api = {
  detectTools: () => invoke("detect_tools"),
  probeVideo: (path) => invoke("probe_video", { path }),
  probeImages: (paths) => invoke("probe_images", { paths }),
  defaultProjectDir: (videoPath) => invoke("default_project_dir", { videoPath }),
  loadProject: (projectDir) => invoke("load_project", { projectDir }),
  sceneFrame: (projectDir) => invoke("scene_frame", { projectDir }),
  cancelJob: () => invoke("cancel_job"),
  splatInfo: (path) => invoke("splat_info", { path }),
  exportSplat: (projectDir, options, onProgress) =>
    invoke<ExportStats>("export_splat", { projectDir, options, onProgress: channel(onProgress) }),
  installPlan: (rmbg) => invoke<InstallPlan>("install_plan", { rmbg }),
  cancelInstall: () => invoke("cancel_install"),
  installTools: (rmbg, onEvent) => invoke("install_tools", { rmbg, onEvent: channel<InstallEvent>(onEvent) }),
  startJob: (config, onEvent) => invoke("start_job", { config, onEvent: channel<JobEvent>(onEvent) }),
};

export const tauriHost: Host = {
  api,
  // Protocolo `asset:` do Tauri.
  fileUrl(path, bust) {
    const url = convertFileSrc(path);
    return bust === undefined ? url : `${url}?v=${bust}`;
  },
  async pickFiles(extensions) {
    const p = await open({ multiple: true, filters: [{ name: "Vídeo ou imagens", extensions }] });
    return p ?? [];
  },
  async pickFramesFolder() {
    const p = await open({ directory: true, multiple: false });
    return typeof p === "string" ? [p] : [];
  },
  async pickFolder(defaultPath) {
    const p = await open({ directory: true, multiple: false, defaultPath });
    return typeof p === "string" ? p : null;
  },
  reveal: (path) => revealItemInDir(path),
  readBlob: (path) => fetch(convertFileSrc(path)).then((r) => r.blob()),
  // O desktop compartilha pela versão web publicada (CORS liberado no Worker).
  shareOrigin: import.meta.env.VITE_SHARE_ORIGIN ?? "https://gausify-orbit.obraiadev.workers.dev",
  onDrop(handler) {
    const un = getCurrentWebview().onDragDropEvent((e) => {
      if (e.payload.type === "drop") handler(e.payload.paths);
    });
    return () => void un.then((f) => f());
  },
};
