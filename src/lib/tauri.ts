// Desktop app host: Rust backend commands via Tauri.

import { Channel, convertFileSrc, invoke } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { PACKAGE_EXT, type Api, type ExportStats, type Host, type InstallEvent, type InstallPlan, type JobEvent } from "./api";

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
  setFramesRejected: (projectDir, names, rejected) => invoke("set_frames_rejected", { projectDir, names, rejected }),
  cancelJob: () => invoke("cancel_job"),
  splatInfo: (path) => invoke("splat_info", { path }),
  exportSplat: (projectDir, options, onProgress) =>
    invoke<ExportStats>("export_splat", { projectDir, options, onProgress: channel(onProgress) }),
  packProject: (request, onProgress) => invoke("pack_project", { request, onProgress: channel(onProgress) }),
  unpackProject: (path, onProgress) => invoke("unpack_project", { path, onProgress: channel(onProgress) }),
  installPlan: (rmbg) => invoke<InstallPlan>("install_plan", { rmbg }),
  cancelInstall: () => invoke("cancel_install"),
  installTools: (rmbg, onEvent) => invoke("install_tools", { rmbg, onEvent: channel<InstallEvent>(onEvent) }),
  startJob: (config, onEvent) => invoke("start_job", { config, onEvent: channel<JobEvent>(onEvent) }),
};

export const tauriHost: Host = {
  api,
  // Tauri's `asset:` protocol.
  fileUrl(path, bust) {
    const url = convertFileSrc(path);
    return bust === undefined ? url : `${url}?v=${bust}`;
  },
  async pickFiles(extensions) {
    const p = await open({ multiple: true, filters: [{ name: "Video or images", extensions }] });
    return p ?? [];
  },
  async pickFramesFolder() {
    // Several folders: one batch item each.
    const p = await open({ directory: true, multiple: true });
    return typeof p === "string" ? [p] : (p ?? []);
  },
  async pickFolder(defaultPath) {
    const p = await open({ directory: true, multiple: false, defaultPath });
    return typeof p === "string" ? p : null;
  },
  async pickPackageDest(projectDir) {
    // Next to the project folder: video_splat → video_splat.gorbit.
    const p = await save({
      defaultPath: `${projectDir.replace(/[\\/]+$/, "")}.${PACKAGE_EXT}`,
      filters: [{ name: "Gausify Orbit", extensions: [PACKAGE_EXT] }],
    });
    return p ?? null;
  },
  reveal: (path) => revealItemInDir(path),
  readBlob: (path) => fetch(convertFileSrc(path)).then((r) => r.blob()),
  // Desktop shares via the published web version (CORS allowed on the Worker).
  shareOrigin: import.meta.env.VITE_SHARE_ORIGIN ?? "https://gausify-orbit.obraiadev.workers.dev",
  onDrop(handler) {
    const un = getCurrentWebview().onDragDropEvent((e) => {
      if (e.payload.type === "drop") handler(e.payload.paths);
    });
    return () => void un.then((f) => f());
  },
};
