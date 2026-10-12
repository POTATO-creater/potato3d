// ZIP do pacote do projeto (.gorbit): ida e volta, e compatibilidade com
// leitores e gravadores de fora (Python zipfile, como o crate `zip` do
// desktop).  pnpm test:web
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { crc32, entryBlob, readZip, ZipWriter } from "./zip.ts";

const bytes = (n: number, seed: number) => Uint8Array.from({ length: n }, (_, i) => (i * seed + (i >> 7)) & 0xff);

async function build(files: Record<string, { data: Uint8Array; deflate: boolean }>) {
  const parts: (Blob | Uint8Array)[] = [];
  const zip = new ZipWriter((p) => void parts.push(p));
  for (const [name, f] of Object.entries(files)) await zip.add(name, new Blob([f.data as BlobPart]), f.deflate);
  const size = await zip.finish();
  const blob = new Blob(parts as BlobPart[]);
  assert.equal(blob.size, size);
  return blob;
}

const python = (() => {
  try {
    execFileSync("python3", ["-c", "import zipfile"]);
    return true;
  } catch {
    return false;
  }
})();

test("crc32 do ZIP", () => {
  assert.equal(crc32(new TextEncoder().encode("123456789")), 0xcbf43926);
  const a = bytes(1000, 7);
  assert.equal(crc32(a.subarray(300), crc32(a.subarray(0, 300))), crc32(a));
});

test("ida e volta: STORE vira fatia, DEFLATE descompacta", async () => {
  const files = {
    "manifest.json": { data: new TextEncoder().encode(JSON.stringify({ ação: "ç" })), deflate: true },
    "dataset/images/00001.jpg": { data: bytes(200_000, 13), deflate: false },
    "dataset/sparse/0/points3D.bin": { data: new Uint8Array(300_000).fill(9), deflate: true },
    "vazio.bin": { data: new Uint8Array(0), deflate: true },
  };
  const blob = await build(files);
  const entries = await readZip(blob);
  assert.deepEqual([...entries.keys()], Object.keys(files));
  for (const [name, f] of Object.entries(files)) {
    const e = entries.get(name)!;
    assert.equal(e.size, f.data.length);
    assert.deepEqual(new Uint8Array(await (await entryBlob(blob, e)).arrayBuffer()), f.data);
  }
  // Compactado de verdade.
  assert.ok(entries.get("dataset/sparse/0/points3D.bin")!.csize < 10_000);
});

test("gravado aqui, lido pelo Python (CRC e tamanhos)", { skip: !python && "sem python3" }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "gorbit-"));
  const path = join(dir, "a.gorbit");
  const img = bytes(70_000, 5);
  const blob = await build({
    "project.json": { data: new TextEncoder().encode('{"frames":[]}'), deflate: true },
    "dataset/images/ação.jpg": { data: img, deflate: false },
  });
  writeFileSync(path, new Uint8Array(await blob.arrayBuffer()));
  const out = execFileSync("python3", [
    "-I",
    "-c",
    "import sys,zipfile,hashlib;z=zipfile.ZipFile(sys.argv[1]);assert z.testzip() is None;print(z.read('project.json').decode());print(hashlib.sha1(z.read('dataset/images/ação.jpg')).hexdigest())",
    path,
  ]).toString();
  const [json, sha] = out.trim().split("\n");
  assert.equal(json, '{"frames":[]}');
  assert.equal(sha, createHash("sha1").update(img).digest("hex"));
});

test("gravado pelo Python, lido aqui (com e sem compressão)", { skip: !python && "sem python3" }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "gorbit-"));
  const path = join(dir, "b.gorbit");
  execFileSync("python3", [
    "-I",
    "-c",
    "import sys,zipfile;z=zipfile.ZipFile(sys.argv[1],'w');z.writestr('manifest.json','{\"version\":1}',zipfile.ZIP_DEFLATED);z.writestr('splat.ply',b'ply'*5000,zipfile.ZIP_DEFLATED);z.writestr('dataset/images/1.jpg',b'\\xff\\xd8jpg',zipfile.ZIP_STORED);z.close()",
    path,
  ]);
  const blob = new Blob([readFileSync(path)]);
  const entries = await readZip(blob);
  assert.equal(await (await entryBlob(blob, entries.get("manifest.json")!)).text(), '{"version":1}');
  assert.equal((await (await entryBlob(blob, entries.get("splat.ply")!)).text()).length, 15000);
  assert.deepEqual(new Uint8Array(await (await entryBlob(blob, entries.get("dataset/images/1.jpg")!)).arrayBuffer()), Uint8Array.from([0xff, 0xd8, 0x6a, 0x70, 0x67]));
});
