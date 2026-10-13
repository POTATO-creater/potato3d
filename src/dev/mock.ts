// UI-only development mode (`pnpm dev:mock`): runs in the browser without
// Tauri, replaying the JSON events recorded by the CLI
// (`cargo run --example cli -- config.json > events.jsonl`).
//
// Variables (in .env.local):
//   VITE_MOCK_VIDEO   path to a test video
//   VITE_MOCK_IMAGES  comma-separated test images (picking files or a folder
//                     opens this set)
//   VITE_MOCK_EVENTS  .jsonl file with the recorded events
//   VITE_MOCK_SCENE   JSON from `cli scene <project>`
//
// `?project=<folder>` opens a real project (project.json); export is
// simulated and returns the `lastExport` saved in it (`cli export <project>`).
import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import type { Channel } from "@tauri-apps/api/core";
import type { JobEvent } from "../lib/api";

// `?events=…&scene=…&video=…` in the URL overrides .env.local (to compare results).
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

/** Header of a .ply served by Vite: count and SH degree. */
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
  if (!project?.lastExport) throw new Error("project has no lastExport (run `cli export` first)");
  return project.lastExport;
}

let cancelled = false;
// `?missing=1` simulates first run (tools missing until "installed").
let toolsInstalled = params.get("missing") !== "1";
// `?rmbg=0`: the RMBG-1.4 model has not been downloaded yet.
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

async function replay(channel: Channel<JobEvent>, from: string, to = "train") {
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
    if (e.type === "finished") return to === "train";
    return !stage || (order.indexOf(stage) >= order.indexOf(from) && order.indexOf(stage) <= order.indexOf(to));
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
  // Full recordings end with training: the end of a partial job is synthetic.
  if (to !== "train") channel.onmessage({ type: "finished", ply: null, seconds: 1 });
}

/** Open project, to simulate frame rejection without touching the disk. */
let project: any = null;

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
          if (!paths.length) throw new Error("no images (jpg, jpeg, png, webp) found");
          return { paths, dir: paths[0].replace(/\/[^/]*$/, ""), width: 960, height: 540 };
        }
        case "load_project":
          return (project =
            (env.project && (await readJson(`${env.project}/project.json`))) ?? {
              videoPath: null,
              frames: [],
              poses: null,
              finalPly: null,
              lastPsnr: null,
            });
        case "set_frames_rejected": {
          // Paths stay where they are: only the flag changes.
          const names = new Set<string>(args.names);
          let changed = false;
          project = {
            ...project,
            frames: project.frames.map((f: any) => {
              if (!!f.rejected === args.rejected || !names.has(f.path.split("/").pop())) return f;
              changed = true;
              return { ...f, rejected: args.rejected || undefined };
            }),
          };
          if (changed) project.poses = null;
          return project;
        }
        case "default_project_dir":
          return env.project ?? args.videoPath.replace(/\.[^.]+$/, "_splat");
        case "splat_info":
          return plyInfo(args.path);
        case "export_splat":
          return fakeExport(args.onProgress, args.projectDir);
        case "scene_frame": {
          const f = await readJson(env.VITE_MOCK_SCENE);
          if (!f) throw new Error("no poses");
          return f;
        }
        case "start_job":
          replay(args.onEvent, args.config.fromStage, args.config.toStage);
          return null;
        case "install_plan":
          if (params.get("os") !== "windows")
            return {
              packages: args.rmbg ? [{ id: "rmbg", label: "RMBG-1.4 masking", file: "model.onnx", size: 176_153_355 }] : [],
              totalBytes: args.rmbg ? 176_153_355 : 0,
              cuda: false,
              note: "On macOS, install via Homebrew: brew install ffmpeg colmap",
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
          if (args.options?.directory) return env.VITE_MOCK_IMAGES ? "/tmp/frames" : "/tmp/project";
          return args.options?.multiple ? [env.VITE_MOCK_IMAGES ? env.VITE_MOCK_IMAGES.split(",")[0] : env.VITE_MOCK_VIDEO] : env.VITE_MOCK_VIDEO;
        case "plugin:opener|reveal_item_in_dir":
          console.info("[mock] reveal", args);
          return null;
        default:
          console.warn("[mock] unmocked command:", cmd, args);
          return null;
      }
    },
    { shouldMockEvents: true },
  );
}
