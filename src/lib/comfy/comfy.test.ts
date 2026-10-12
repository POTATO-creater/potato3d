import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { normalizeUrl, outputFiles, wsUrl } from "./client.ts";
import { initialProgress, progressFraction, reduceProgress } from "./progress.ts";
import { checkIO, effectiveIO, guessIO, listNodes, parseWorkflow, withInput, WorkflowParseError } from "./workflow.ts";

const ltx = readFileSync(new URL("./fixtures/ltx-restore.json", import.meta.url), "utf8");

test("workflow LTX: acha LoadVideo.file e SaveVideo", () => {
  const g = parseWorkflow(ltx);
  assert.deepEqual(guessIO(g), { input: { node: "5001", field: "file" }, output: { node: "4852" } });
  assert.deepEqual(checkIO(g, guessIO(g)!), []);
  const n = listNodes(g).find((x) => x.id === "5001")!;
  assert.equal(n.title, "Load Video");
  // ligações não aparecem como campos editáveis
  assert.ok(!listNodes(g).find((x) => x.id === "4852")!.literalInputs.some((i) => i.name === "video"));
});

test("withInput troca só o campo de entrada, sem mexer no original", () => {
  const g = parseWorkflow(ltx);
  const io = guessIO(g)!;
  const out = withInput(g, io, "retoque_abc.mp4");
  assert.equal(out["5001"].inputs.file, "retoque_abc.mp4");
  assert.equal(g["5001"].inputs.file, "Gravando 2026-10-12 173208.mp4");
});

test("duração: palpite no EmptyAudio (segundos), fps do CreateVideo, e injeção", () => {
  const g = parseWorkflow(ltx);
  const io = effectiveIO(g, guessIO(g)!);
  assert.deepEqual(io.length, { node: "9017", field: "duration", unit: "seconds" });
  assert.equal(io.fps, 30);
  assert.deepEqual(checkIO(g, io), []);
  const out = withInput(g, io, "a.mp4", { frames: 97, fps: 30 });
  assert.equal(out["9017"].inputs.duration, 3.23);
  const frames = withInput(g, { ...io, length: { node: "9017", field: "duration", unit: "frames" } }, "a.mp4", { frames: 97, fps: 30 });
  assert.equal(frames["9017"].inputs.duration, 97);
  // null = sem campo de duração: nada muda além do arquivo
  assert.equal(withInput(g, { ...io, length: null }, "a.mp4", { frames: 97, fps: 30 })["9017"].inputs.duration, 4);
  assert.deepEqual(checkIO(g, { ...io, length: { node: "9018", field: "codec", unit: "seconds" } }), ["lengthField"]);
});

test("formato do editor e lixo são recusados com o motivo", () => {
  const reason = (s: string) => {
    try {
      parseWorkflow(s);
      return null;
    } catch (e) {
      return (e as WorkflowParseError).reason;
    }
  };
  assert.equal(reason(JSON.stringify({ nodes: [], links: [], version: 0.4 })), "editorFormat");
  assert.equal(reason("{"), "notJson");
  assert.equal(reason("{}"), "empty");
  assert.equal(reason(JSON.stringify({ a: { inputs: {} } })), "notWorkflow");
});

test("checkIO aponta nó ou campo inexistente e campo que é ligação", () => {
  const g = parseWorkflow(ltx);
  assert.deepEqual(checkIO(g, { input: { node: "x", field: "file" }, output: { node: "4852" } }), ["inputNode"]);
  assert.deepEqual(checkIO(g, { input: { node: "9020", field: "video" }, output: { node: "y" } }), ["inputField", "outputNode"]);
});

test("progresso: cache, nó atual com passo, sucesso", () => {
  const id = "p1";
  const msgs = [
    { type: "status", data: { status: { exec_info: { queue_remaining: 2 } } } },
    { type: "execution_start", data: { prompt_id: id } },
    { type: "execution_cached", data: { prompt_id: id, nodes: ["a", "b"] } },
    { type: "executing", data: { prompt_id: id, node: "c" } },
    { type: "progress", data: { prompt_id: id, node: "c", value: 4, max: 8 } },
    { type: "executing", data: { prompt_id: "outro", node: "z" } },
  ];
  let p = initialProgress(id, 10);
  p = reduceProgress(p, msgs[0]);
  assert.equal(p.queue, 2);
  for (const m of msgs.slice(1)) p = reduceProgress(p, m);
  assert.equal(p.status, "running");
  assert.equal(p.node, "c");
  assert.deepEqual(p.done.sort(), ["a", "b"]);
  assert.equal(progressFraction(p), 0.25); // (2 + 0,5) / 10
  p = reduceProgress(p, { type: "executing", data: { prompt_id: id, node: "d" } });
  assert.deepEqual(p.done.sort(), ["a", "b", "c"]);
  p = reduceProgress(p, { type: "execution_success", data: { prompt_id: id } });
  assert.equal(p.status, "success");
  assert.equal(progressFraction(p), 1);
  // depois do fim, nada muda
  assert.equal(reduceProgress(p, { type: "executing", data: { prompt_id: id, node: "e" } }), p);
});

test("progresso: erro guarda nó e mensagem", () => {
  let p = initialProgress("p", 3);
  p = reduceProgress(p, {
    type: "execution_error",
    data: { prompt_id: "p", node_id: "5004:5602", node_type: "UNETLoader", exception_message: "Value not in list\n" },
  });
  assert.equal(p.status, "error");
  assert.deepEqual(p.error, { node: "5004:5602", nodeType: "UNETLoader", message: "Value not in list" });
});

test("URL, WebSocket e arquivos de saída", () => {
  assert.equal(normalizeUrl(" 127.0.0.1:8188/ "), "http://127.0.0.1:8188");
  assert.equal(normalizeUrl("https://gpu.exemplo.com/comfy/"), "https://gpu.exemplo.com/comfy");
  assert.equal(wsUrl("https://h", "a b"), "wss://h/ws?clientId=a%20b");
  assert.deepEqual(outputFiles({ images: [{ filename: "output_00001_.mp4", subfolder: "", type: "output" }], animated: [true] }), [
    { filename: "output_00001_.mp4", subfolder: "", type: "output" },
  ]);
  assert.deepEqual(outputFiles({ gifs: [{ filename: "a.mp4" }] }), [{ filename: "a.mp4", subfolder: "", type: "output" }]);
});
