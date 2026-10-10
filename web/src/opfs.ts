// Datasets copiados para o OPFS (armazenamento privado da origem): dá ao Brush
// um FileSystemDirectoryHandle mesmo sem `showDirectoryPicker` — vindo de um
// <input webkitdirectory> ou de uma URL.

const MODEL_FILES = ["cameras.bin", "images.bin", "points3D.bin"];

/** Pasta nova em `<group>/<carimbo>/<name>`, apagando as anteriores do grupo. */
export async function freshDir(name: string, keep?: FileSystemDirectoryHandle, group = "datasets") {
  const root = await navigator.storage.getDirectory();
  const datasets = await root.getDirectoryHandle(group, { create: true });
  // `keep` pode estar no próprio OPFS (ex.: a origem de uma cópia).
  const keepSlot = keep ? (await root.resolve(keep))?.[1] : undefined;
  // Um dataset por vez: os anteriores saem para não acumular GBs na origem.
  // Um que ainda esteja aberto (ex.: treino em andamento) fica para a próxima.
  for await (const key of datasets.keys()) {
    if (key === keepSlot) continue;
    await datasets.removeEntry(key, { recursive: true }).catch(() => {});
  }
  const slot = await datasets.getDirectoryHandle(String(Date.now()), { create: true });
  return slot.getDirectoryHandle(name, { create: true });
}

async function writeFile(dir: FileSystemDirectoryHandle, path: string[], data: Blob) {
  let d = dir;
  for (const part of path.slice(0, -1)) d = await d.getDirectoryHandle(part, { create: true });
  const w = await (await d.getFileHandle(path[path.length - 1], { create: true })).createWritable();
  await w.write(data);
  await w.close();
}

/** Arquivos de um <input webkitdirectory>; o primeiro nível do caminho é a
 *  própria pasta escolhida. */
export async function datasetFromFiles(files: File[], onProgress: (done: number, total: number) => void) {
  const name = files[0]?.webkitRelativePath.split("/")[0] || "dataset";
  const dir = await freshDir(name);
  for (let i = 0; i < files.length; i++) {
    const path = files[i].webkitRelativePath.split("/").slice(1);
    if (path.length && !path.some((p) => p.startsWith("."))) await writeFile(dir, path, files[i]);
    onProgress(i + 1, files.length);
  }
  return dir;
}

/** Dataset COLMAP servido por HTTP (`<base>/images/…` e `<base>/sparse/0/*.bin`).
 *  Sem listagem de diretório: os nomes das imagens vêm do `images.bin`. */
export async function datasetFromUrl(base: string, onProgress: (done: number, total: number) => void) {
  base = base.replace(/\/+$/, "");
  const get = async (rel: string) => {
    const r = await fetch(`${base}/${rel}`);
    if (!r.ok) throw new Error(`${r.status} ao baixar ${base}/${rel}`);
    return r.blob();
  };
  const model = await Promise.all(MODEL_FILES.map((f) => get(`sparse/0/${f}`)));
  const names = imageNames(await model[1].arrayBuffer());
  const dir = await freshDir(base.split("/").pop() || "dataset");
  for (let i = 0; i < MODEL_FILES.length; i++) await writeFile(dir, ["sparse", "0", MODEL_FILES[i]], model[i]);
  let done = 0;
  const queue = [...names];
  const worker = async () => {
    for (let n = queue.shift(); n !== undefined; n = queue.shift()) {
      await writeFile(dir, ["images", ...n.split("/")], await get(`images/${n.split("/").map(encodeURIComponent).join("/")}`));
      onProgress(++done, names.length);
    }
  };
  await Promise.all(Array.from({ length: 6 }, worker));
  return dir;
}

function imageNames(buf: ArrayBuffer): string[] {
  const v = new DataView(buf);
  const bytes = new Uint8Array(buf);
  let o = 8;
  const n = Number(v.getBigUint64(0, true));
  const names: string[] = [];
  for (let i = 0; i < n; i++) {
    o += 4 + 7 * 8 + 4; // id, qvec, tvec, camera_id
    const end = bytes.indexOf(0, o);
    names.push(new TextDecoder().decode(bytes.subarray(o, end)));
    o = end + 1;
    o += 8 + Number(v.getBigUint64(o, true)) * 24;
  }
  return names;
}

/** Cópia só de `images/` para o OPFS: base para calcular poses sem tocar na
 *  pasta do usuário (o SfM grava `sparse/0` na cópia). */
export async function copyImages(src: FileSystemDirectoryHandle, onProgress: (done: number, total: number) => void) {
  const images = await src.getDirectoryHandle("images");
  const files: File[] = [];
  for await (const [name, h] of images.entries())
    if (h.kind === "file" && /\.(jpe?g|png|webp)$/i.test(name)) files.push(await (h as FileSystemFileHandle).getFile());
  const dir = await freshDir(src.name, src);
  for (let i = 0; i < files.length; i++) {
    await writeFile(dir, ["images", files[i].name], files[i]);
    onProgress(i + 1, files.length);
  }
  return dir;
}
