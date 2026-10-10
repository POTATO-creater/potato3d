// Caminhos da versão web. A interface trata caminhos como texto opaco, como
// no desktop; aqui eles apontam para:
//   /projects/<nome>/…  pastas e arquivos no OPFS (armazenamento da origem)
//   /videos/<nome>      vídeos abertos nesta aba (só em memória)
// e `fileUrl` precisa ser síncrono, então cada arquivo que a interface vai
// exibir é "publicado" antes: ganha uma URL blob: guardada num mapa.

import type { VideoInfo } from "../../../src/lib/api";

export const PROJECTS = "/projects";

const videos = new Map<string, { file: File; info?: VideoInfo }>();
const urls = new Map<string, string>();

export const basename = (p: string) => p.split("/").pop() ?? p;
export const stem = (name: string) => basename(name).replace(/\.[^.]+$/, "");
const parts = (path: string) => path.split("/").filter(Boolean);

export function addVideo(file: File): string {
  const path = `/videos/${file.name}`;
  videos.set(path, { file });
  publish(path, file);
  return path;
}

export function video(path: string) {
  const v = videos.get(path);
  if (!v) throw new Error("O vídeo não está aberto nesta aba: arraste-o para a janela de novo.");
  return v;
}

/** URL do arquivo publicado, ou "" se ainda não foi. */
export const urlOf = (path: string) => urls.get(path) ?? "";

/** Dá ao arquivo uma URL estável até a próxima publicação do mesmo caminho.
 *  A antiga é solta mais tarde: pode haver uma imagem ou splat carregando. */
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
    /* pasta do caminho não existe */
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
    /* já não existe */
  }
}

/** Arquivos de uma pasta (só o primeiro nível), em ordem de nome. */
export async function listFiles(path: string, pattern?: RegExp): Promise<string[]> {
  const names: string[] = [];
  try {
    for await (const [name, h] of (await dirAt(path)).entries())
      if (h.kind === "file" && (!pattern || pattern.test(name))) names.push(name);
  } catch {
    /* pasta não existe */
  }
  return names.sort();
}

/** Publica todos os arquivos (do primeiro nível) de uma pasta. */
export async function publishDir(path: string, pattern?: RegExp) {
  const d = await dirAt(path).catch(() => null);
  if (!d) return;
  for await (const [name, h] of d.entries())
    if (h.kind === "file" && (!pattern || pattern.test(name))) publish(`${path}/${name}`, await (h as FileSystemFileHandle).getFile());
}

/** Lê um arquivo publicado (inclusive os que só existem em memória). */
export async function readPublished(path: string): Promise<Blob> {
  const url = urls.get(path);
  if (url) return (await fetch(url)).blob();
  return readFile(path);
}
