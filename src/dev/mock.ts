// Modo de desenvolvimento só da interface (`pnpm dev:mock`): roda no navegador,
// sem Tauri, reproduzindo os eventos JSON gravados pela CLI
// (`cargo run --example cli -- config.json > eventos.jsonl`).
//
// Variáveis (em .env.local):
//   VITE_MOCK_VIDEO   caminho de um vídeo de teste
//   VITE_MOCK_IMAGES  imagens de teste separadas por vírgula (escolher arquivos
//                     ou pasta abre esse conjunto)
//   VITE_MOCK_EVENTS  arquivo .jsonl com os eventos gravados
//   VITE_MOCK_SCENE   JSON de `cli scene <projeto>`
//
// `?project=<pasta>` abre um projeto real (project.json); a exportação é
// simulada e devolve o `lastExport` gravado nele (`cli export <projeto>`).
import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import type { Channel } from "@tauri-apps/api/core";
import type { JobEvent } from "../lib/api";

// `?events=…&scene=…&video=…` na URL sobrepõe o .env.local (comparar resultados).
const params = new URLSearchParams(location.search);
const env = {
  VITE_MOCK_VIDEO: params.get("video") ?? import.meta.env.VITE_MOCK_VIDEO,
  VITE_MOCK_IMAGES: params.get("images") ?? import.meta.env.VITE_MOCK_IMAGES,
  VITE_MOCK_EVENTS: params.get("events") ?? import.meta.env.VITE_MOCK_EVENTS,
  VITE_MOCK_SCENE: params.get("scene") ?? import.meta.env.VITE_MOCK_SCENE,
  project: params.get("project"),
};
const fsUrl = (p: string) => `/@fs${p}`;

async function readJson<T>(path: string | undefined): Promise<T | null> {
  if (!path) return null;
  const r = await fetch(fsUrl(path));
  return r.ok ? r.json() : null;
}

async function readEvents(): Promise<JobEvent[]> {
  if (!env.VITE_MOCK_EVENTS) return [];
  const text = await (await fetch(fsUrl(env.VITE_MOCK_EVENTS))).text();
  return text
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

/** Cabeçalho de um .ply servido pelo Vite: contagem e grau SH. */
async function plyInfo(path: string) {
  const r = await fetch(fsUrl(path));
  const bytes = new Uint8Array(await r.arrayBuffer());
  const head = new TextDecoder().decode(bytes.subarray(0, 4096));
  const count = Number(/element vertex (\d+)/.exec(head)?.[1] ?? 0);
  const rest = (head.match(/property float f_rest_/g) ?? []).length;
  return { count, shDegree: [0, 9, 24, 45].indexOf(rest), bytes: bytes.length };
}

async function fakeExport(channel: Channel<any>, projectDir: string) {
  const project = await readJson<any>(`${projectDir}/project.json`);
  const steps = ["export.readingPly", "export.merging", "export.clustering", "export.compressing"];
  for (let i = 0; i <= 20; i++) {
    channel.onmessage({ fraction: i / 20, message: { key: steps[Math.min(steps.length - 1, Math.floor(i / 6))] } });
    await new Promise((r) => setTimeout(r, 70));
  }
  if (!project?.lastExport) throw new Error("projeto sem lastExport (rode `cli export` antes)");
  return project.lastExport;
}

let cancelled = false;
// `?missing=1` simula o primeiro uso (ferramentas ausentes até "instalar").
let toolsInstalled = params.get("missing") !== "1";
// `?rmbg=0`: o modelo RMBG-1.4 ainda não foi baixado.
let rmbgInstalled = params.get("rmbg") !== "0";

async function fakeInstall(channel: Channel<any>) {
  cancelled = false;
  const pkgs = [
    { id: "colmap", label: "COLMAP 4.2.1 (CUDA)", total: 414_666_626 },
    { id: "ffmpeg", label: "ffmpeg", total: 171_473_550 },
  ];
  for (const p of pkgs) {
    channel.onmessage({ type: "started", ...p });
    for (let done = 0; done <= p.total; done += p.total / 25) {
      if (cancelled) return channel.onmessage({ type: "cancelled" });
      channel.onmessage({ type: "progress", id: p.id, done, total: p.total });
      await new Promise((r) => setTimeout(r, 60));
    }
    channel.onmessage({ type: "extracting", id: p.id });
    await new Promise((r) => setTimeout(r, 700));
    channel.onmessage({ type: "done", id: p.id });
  }
  toolsInstalled = true;
  rmbgInstalled = true;
  channel.onmessage({ type: "finished" });
}

async function replay(channel: Channel<JobEvent>, from: string) {
  cancelled = false;
  const order = ["frames", "masks", "poses", "train"];
  const events = (await readEvents()).filter((e) => {
    const stage =
      "stage" in e
        ? e.stage
        : e.type === "framesReady"
          ? "frames"
          : e.type === "masksReady"
            ? "masks"
            : e.type === "posesReady"
              ? "poses"
              : "train";
    return !stage || order.indexOf(stage) >= order.indexOf(from);
  });
  for (const e of events) {
    if (cancelled) {
      channel.onmessage({ type: "cancelled" });
      return;
    }
    const delay = e.type === "log" ? 4 : e.type === "train" ? 220 : e.type === "progress" ? 40 : 300;
    await new Promise((r) => setTimeout(r, delay));
    channel.onmessage(e);
  }
}

export function installMock() {
  mockWindows("main");
  (window as any).__TAURI_INTERNALS__.convertFileSrc = (p: string) => fsUrl(p);

  mockIPC(
    async (cmd, args: any) => {
      switch (cmd) {
        case "detect_tools":
          return {
            ffmpeg: toolsInstalled ? "/opt/homebrew/bin/ffmpeg" : null,
            ffprobe: toolsInstalled ? "/opt/homebrew/bin/ffprobe" : null,
            colmap: toolsInstalled ? "/opt/homebrew/bin/colmap" : null,
            colmapCuda: params.get("os") === "windows",
            brush: "mock",
            masker: params.get("os") === "windows" ? null : "mock",
            birefnetModel: params.get("os") === "windows" && toolsInstalled ? "mock" : null,
            modnetModel: params.get("os") === "windows" && toolsInstalled ? "mock" : null,
            rmbgModel: rmbgInstalled ? "mock" : null,
            os: params.get("os") ?? "macos",
          };
        case "probe_video":
          return { width: 960, height: 540, fps: 24, duration: 10, codec: "h264" };
        case "probe_images": {
          const paths = (env.VITE_MOCK_IMAGES?.split(",") ?? args.paths).filter((p: string) => /\.(jpe?g|png|webp)$/i.test(p));
          if (!paths.length) throw new Error("nenhuma imagem (jpg, jpeg, png, webp) encontrada");
          return { paths, dir: paths[0].replace(/\/[^/]*$/, ""), width: 960, height: 540 };
        }
        case "load_project":
          return (
            (env.project && (await readJson(`${env.project}/project.json`))) ?? {
              videoPath: null,
              frames: [],
              poses: null,
              finalPly: null,
              lastPsnr: null,
            }
          );
        case "default_project_dir":
          return env.project ?? args.videoPath.replace(/\.[^.]+$/, "_splat");
        case "splat_info":
          return plyInfo(args.path);
        case "export_splat":
          return fakeExport(args.onProgress, args.projectDir);
        case "scene_frame": {
          const f = await readJson(env.VITE_MOCK_SCENE);
          if (!f) throw new Error("sem poses");
          return f;
        }
        case "start_job":
          replay(args.onEvent, args.config.fromStage);
          return null;
        case "install_plan":
          if (params.get("os") !== "windows")
            return {
              packages: args.rmbg ? [{ id: "rmbg", label: "Recorte RMBG-1.4", file: "model.onnx", size: 176_153_355 }] : [],
              totalBytes: args.rmbg ? 176_153_355 : 0,
              cuda: false,
              note: "No macOS instale pelo Homebrew: brew install ffmpeg colmap",
            };
          return {
            packages: [
              { id: "colmap", label: "COLMAP 4.2.1 (CUDA)", file: "colmap-x64-windows-cuda.zip", size: 414_666_626 },
              { id: "ffmpeg", label: "ffmpeg", file: "ffmpeg-n9.0-latest-win64-lgpl-9.0.zip", size: 171_473_550 },
            ],
            totalBytes: 586_140_176,
            cuda: true,
            note: null,
          };
        case "install_tools":
          fakeInstall(args.onEvent);
          return null;
        case "cancel_install":
          cancelled = true;
          return null;
        case "cancel_job":
          cancelled = true;
          return null;
        case "plugin:dialog|open":
          if (args.options?.directory) return env.VITE_MOCK_IMAGES ? "/tmp/frames" : "/tmp/projeto";
          return args.options?.multiple ? [env.VITE_MOCK_IMAGES ? env.VITE_MOCK_IMAGES.split(",")[0] : env.VITE_MOCK_VIDEO] : env.VITE_MOCK_VIDEO;
        case "plugin:opener|reveal_item_in_dir":
          console.info("[mock] revelar", args);
          return null;
        default:
          console.warn("[mock] comando sem mock:", cmd, args);
          return null;
      }
    },
    { shouldMockEvents: true },
  );
}
