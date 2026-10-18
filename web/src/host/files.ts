// Web paths. The UI treats paths as opaque strings; here they point to:
//   /projects/<name>/…  folders and files in OPFS (origin storage)
//   /videos/<name>      videos opened in this tab (in memory only)
//   /inputs/<folder>/…  images used as frames, likewise
// and `fileUrl` must be synchronous, so every file the UI will display is
// "published" first: it gets a blob: URL kept in a map.

import type { VideoInfo } from "../../../src/lib/api";

export const PROJECTS = "/projects";

const videos = new Map<string, { file: File; info?: VideoInfo }>();
const images = new Map<string, File>();
/** Project packages (.gorbit) opened in this tab. */
const packages = new Map<string, File>();
const urls = new Map<string, string>();

export const basename = (p: string) => p.split("/").pop() ?? p;
export const stem = (name: string) => basename(name).replace(/\.[^.]+$/, "");
const parts = (path: string) => path.split("/").filter(Boolean);

const sameFile = (a: File, b: File) => a.name === b.name && a.size === b.size && a.lastModified === b.lastModified;

/** The same file again keeps its path; another one with the same name (from
 *  another folder) gets its own, and so its own project. */
export function addVideo(file: File): string {
  const ext = file.name.includes(".") ? file.name.slice(file.name.lastIndexOf(".")) : "";
  let path = `/videos/${file.name}`;
  for (let n = 2; videos.has(path) && !sameFile(videos.get(path)!.file, file); n++) path = `/videos/${stem(file.name)} (${n})${ext}`;
  if (!videos.has(path)) {
    videos.set(path, { file });
    publish(path, file);
  }
  return path;
}

export function video(path: string) {
  const v = videos.get(path);
  if (!v) throw new Error("The video is not open in this tab: drag it onto the window again.");
  return v;
}

export function addPackage(file: File): string {
  const path = `/packages/${file.name}`;
  packages.set(path, file);
  return path;
}

export function packageFile(path: string): File {
  const f = packages.get(path);
  if (!f) throw new Error("The package is not open in this tab: drag it onto the window again.");
  return f;
}

/** Images from a folder (or loose ones, grouped under `name`) to use as
 *  frames. The same images again replace the set; a different set with the
 *  same name (another folder) gets its own. */
export function addImages(files: File[], name: string): string[] {
  const base = `/inputs/${name.replace(/[\\/]/g, "_") || "images"}`;
  const key = (fs: File[]) => fs.map((f) => `${f.name}:${f.size}:${f.lastModified}`).sort().join("|");
  let dir = base;
  for (let n = 2; ; n++) {
    const open = [...images].filter(([p]) => p.startsWith(`${dir}/`)).map(([, f]) => f);
    if (!open.length || key(open) === key(files)) break;
    dir = `${base} (${n})`;
  }
  for (const p of images.keys()) if (p.startsWith(`${dir}/`)) images.delete(p);
  unpublishUnder(dir);
  return files.map((file) => {
    const path = `${dir}/${file.name}`;
    images.set(path, file);
    publish(path, file);
    return path;
  });
}

export function inputImage(path: string): File {
  const f = images.get(path);
  if (!f) throw new Error("The images are not open in this tab: drag them onto the window again.");
  return f;
}

/** Name for a group of loose images: their common prefix ("frame_0001.png",
 *  "frame_0002.png" → "frame"). */
export function groupName(names: string[]): string {
  let prefix = names[0] ?? "";
  for (const n of names) while (!n.startsWith(prefix)) prefix = prefix.slice(0, -1);
  return prefix.replace(/[\d\s._-]+$/, "") || "images";
}

/** URL of the published file, or "" if it hasn't been published yet. */
export const urlOf = (path: string) => urls.get(path) ?? "";

/** Gives the file a URL that stays stable until the same path is published again.
 *  The old one is revoked later: an image or splat may still be loading from it. */
export function publish(path: string, blob: Blob) {
  const old = urls.get(path);
  urls.set(path, URL.createObjectURL(blob));
  if (old) setTimeout(() => URL.revokeObjectURL(old), 60_000);
}

export function unpublishUnder(prefix: string) {
  for (const [path, url] of urls)
    if (path === prefix || path.startsWith(`${prefix}/`)) {
      urls.delete(path);
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    }
}

export async function dirAt(path: string, create = false): Promise<FileSystemDirectoryHandle> {
  let d = await navigator.storage.getDirectory();
  for (const p of parts(path)) d = await d.getDirectoryHandle(p, { create });
  return d;
}

export async function exists(path: string) {
  try {
    const ps = parts(path);
    const d = await dirAt(ps.slice(0, -1).join("/"));
    for await (const name of d.keys()) if (name === ps[ps.length - 1]) return true;
  } catch {
    /* parent folder doesn't exist */
  }
  return false;
}

export async function readFile(path: string): Promise<File> {
  const ps = parts(path);
  const d = await dirAt(ps.slice(0, -1).join("/"));
  return (await d.getFileHandle(ps[ps.length - 1])).getFile();
}

export async function writeFile(path: string, data: Blob | BufferSource | string): Promise<File> {
  const ps = parts(path);
  const d = await dirAt(ps.slice(0, -1).join("/"), true);
  const h = await d.getFileHandle(ps[ps.length - 1], { create: true });
  const w = await h.createWritable();
  await w.write(data);
  await w.close();
  return h.getFile();
}

export async function remove(path: string) {
  unpublishUnder(path);
  const ps = parts(path);
  try {
    const d = await dirAt(ps.slice(0, -1).join("/"));
    await d.removeEntry(ps[ps.length - 1], { recursive: true });
  } catch {
    /* already gone */
  }
}

/** Files in a folder (top level only), sorted by name. */
export async function listFiles(path: string, pattern?: RegExp): Promise<string[]> {
  const names: string[] = [];
  try {
    for await (const [name, h] of (await dirAt(path)).entries())
      if (h.kind === "file" && (!pattern || pattern.test(name))) names.push(name);
  } catch {
    /* folder doesn't exist */
  }
  return names.sort();
}

/** Publishes every (top-level) file in a folder. */
export async function publishDir(path: string, pattern?: RegExp) {
  const d = await dirAt(path).catch(() => null);
  if (!d) return;
  for await (const [name, h] of d.entries())
    if (h.kind === "file" && (!pattern || pattern.test(name))) publish(`${path}/${name}`, await (h as FileSystemFileHandle).getFile());
}

/** Reads a published file (including ones that only exist in memory). */
export async function readPublished(path: string): Promise<Blob> {
  const url = urls.get(path);
  if (url) return (await fetch(url)).blob();
  return readFile(path);
}

/** Hands a file to the user as a download. */
export function download(blob: Blob, name: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
}

/** Asks the browser to keep this site's storage under disk pressure. Without
 *  it ("best effort"), Chrome may clear the whole origin while a long job is
 *  running on a nearly full disk. Usually granted to an installed app; a
 *  refusal is silent. */
export function requestPersistence() {
  navigator.storage?.persist?.().catch(() => false);
}

/** A project file vanished mid-job: in practice the browser cleared the site's
 *  storage (low disk space). */
export function storageLost(e: unknown) {
  if (!(e instanceof DOMException && e.name === "NotFoundError")) return e;
  console.error("file missing during the job:", e);
  return new Error("the browser cleared this site's storage during the job (low disk space?); free some space and open the project again");
}
