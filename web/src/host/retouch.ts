// Retoque na versão web: clipes (trajetória renderizada do splat e o vídeo
// restaurado pelo ComfyUI) guardados no OPFS e o treino incremental.

import type { ProjectState, RetouchApi, RetouchClip } from "../../../src/lib/api";
import { listFiles, publish, readFile, remove, writeFile, dirAt } from "./files";
import { layout, readState, saveState } from "./project";

const clipDir = (root: string, id: string) => `${layout(root).retouch}/${id}`;

/** Ids viram nomes de pasta: só letras, dígitos e hífen. */
const safeId = (id: string) => {
  if (!/^[\w-]+$/.test(id)) throw new Error(`id de clipe inválido: ${id}`);
  return id;
};

async function readClip(root: string, id: string): Promise<RetouchClip | null> {
  try {
    const clip = JSON.parse(await (await readFile(`${clipDir(root, id)}/clip.json`)).text()) as RetouchClip;
    // Os vídeos ganham URL para os <video> da revisão.
    for (const p of [clip.reference, clip.restored]) if (p) publish(p, await readFile(p));
    return clip;
  } catch {
    return null;
  }
}

async function clipIds(root: string): Promise<string[]> {
  const ids: string[] = [];
  try {
    for await (const [name, h] of (await dirAt(layout(root).retouch)).entries()) if (h.kind === "directory") ids.push(name);
  } catch {
    /* ainda sem clipes */
  }
  return ids;
}

let applying = null as AbortController | null;

export const retouchApi: RetouchApi = {
  async listClips(projectDir) {
    const clips = await Promise.all((await clipIds(projectDir)).map((id) => readClip(projectDir, id)));
    return clips.filter((c): c is RetouchClip => !!c).sort((a, b) => a.createdAt - b.createdAt);
  },

  async saveClip(projectDir, clip, files) {
    const dir = clipDir(projectDir, safeId(clip.id));
    const saved = { ...clip };
    for (const [rel, blob] of Object.entries(files)) {
      if (rel.includes("..")) throw new Error(`caminho inválido: ${rel}`);
      const path = `${dir}/${rel}`;
      const file = await writeFile(path, blob);
      if (rel === "reference.mp4") saved.reference = path;
      if (rel === "restored.mp4") saved.restored = path;
      if (rel.endsWith(".mp4")) publish(path, file);
    }
    await writeFile(`${dir}/clip.json`, JSON.stringify(saved));
    return saved;
  },

  async deleteClip(projectDir, id) {
    await remove(clipDir(projectDir, safeId(id)));
  },

  async apply() {
    throw new Error("treino incremental ainda não implementado");
  },

  async cancel() {
    applying?.abort();
  },

  async undo(projectDir): Promise<ProjectState> {
    const l = layout(projectDir);
    const state = await readState(projectDir);
    if (!state.retouch) return state;
    const base = await readFile(l.retouchBase);
    publish(l.finalPly, await writeFile(l.finalPly, base));
    await remove(l.retouchBase);
    const next: ProjectState = { ...state, retouch: null, lastExport: null };
    await saveState(projectDir, next);
    return next;
  },
};

/** Frames alpha de um clipe, em ordem. */
export const clipAlphaFiles = (root: string, id: string) =>
  listFiles(`${clipDir(root, id)}/alpha`, /^\d+\.png$/).then((names) => names.map((n) => `${clipDir(root, id)}/alpha/${n}`));
