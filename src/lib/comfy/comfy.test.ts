import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { normalizeUrl, outputFiles, wsUrl } from "./client.ts";
import { initialProgress, progressFraction, reduceProgress } from "./progress.ts";
import { checkIO, guessIO, listNodes, parseWorkflow, withInput, withParams, workflowFps, WorkflowParseError, paramDefault, editableFields } from "./workflow.ts";

const ltx = readFileSync(new URL("./fixtures/ltx-restore.json", import.meta.url), "utf8");

test("LTX workflow: finds LoadVideo.file and SaveVideo", () => {
  const g = parseWorkflow(ltx);
  assert.deepEqual(guessIO(g), { input: { node: "5001", field: "file" }, output: { node: "4852" } });
  assert.deepEqual(checkIO(g, guessIO(g)!), []);
  const n = listNodes(g).find((x) => x.id === "5001")!;
  assert.equal(n.title, "Load Video");
  // links don't show up as editable fields
  assert.ok(!listNodes(g).find((x) => x.id === "4852")!.literalInputs.some((i) => i.name === "video"));
});

test("withInput changes only the input field, leaving the original intact", () => {
  const g = parseWorkflow(ltx);
  const io = guessIO(g)!;
  const out = withInput(g, io, "retouch_abc.mp4");
  assert.equal(out["5001"].inputs.file, "retouch_abc.mp4");
  assert.equal(g["5001"].inputs.file, "Gravando 2026-10-12 173208.mp4");
});

test("reference video fps: the workflow's CreateVideo value, else 24", () => {
  assert.equal(workflowFps(parseWorkflow(ltx)), 30);
  assert.equal(workflowFps({ a: { class_type: "LoadVideo", inputs: { file: "x.mp4" } } }), 24);
});

test("editor format and garbage are rejected with a reason", () => {
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

test("checkIO flags missing node or field and a field that is a link", () => {
  const g = parseWorkflow(ltx);
  assert.deepEqual(checkIO(g, { input: { node: "x", field: "file" }, output: { node: "4852" } }), ["inputNode"]);
  assert.deepEqual(checkIO(g, { input: { node: "9020", field: "video" }, output: { node: "y" } }), ["inputField", "outputNode"]);
});

test("progress: cache, current node with step, success", () => {
  const id = "p1";
  const msgs = [
    { type: "status", data: { status: { exec_info: { queue_remaining: 2 } } } },
    { type: "execution_start", data: { prompt_id: id } },
    { type: "execution_cached", data: { prompt_id: id, nodes: ["a", "b"] } },
    { type: "executing", data: { prompt_id: id, node: "c" } },
    { type: "progress", data: { prompt_id: id, node: "c", value: 4, max: 8 } },
    { type: "executing", data: { prompt_id: "other", node: "z" } },
  ];
  let p = initialProgress(id, 10);
  p = reduceProgress(p, msgs[0]);
  assert.equal(p.queue, 2);
  for (const m of msgs.slice(1)) p = reduceProgress(p, m);
  assert.equal(p.status, "running");
  assert.equal(p.node, "c");
  assert.deepEqual(p.done.sort(), ["a", "b"]);
  assert.equal(progressFraction(p), 0.25); // (2 + 0.5) / 10
  p = reduceProgress(p, { type: "executing", data: { prompt_id: id, node: "d" } });
  assert.deepEqual(p.done.sort(), ["a", "b", "c"]);
  p = reduceProgress(p, { type: "execution_success", data: { prompt_id: id } });
  assert.equal(p.status, "success");
  assert.equal(progressFraction(p), 1);
  // after the end, nothing changes
  assert.equal(reduceProgress(p, { type: "executing", data: { prompt_id: id, node: "e" } }), p);
});

test("progress: error keeps node and message", () => {
  let p = initialProgress("p", 3);
  p = reduceProgress(p, {
    type: "execution_error",
    data: { prompt_id: "p", node_id: "5004:5602", node_type: "UNETLoader", exception_message: "Value not in list\n" },
  });
  assert.equal(p.status, "error");
  assert.deepEqual(p.error, { node: "5004:5602", nodeType: "UNETLoader", message: "Value not in list" });
});

test("URL, WebSocket and output files", () => {
  assert.equal(normalizeUrl(" 127.0.0.1:8188/ "), "http://127.0.0.1:8188");
  assert.equal(normalizeUrl("https://gpu.example.com/comfy/"), "https://gpu.example.com/comfy");
  assert.equal(wsUrl("https://h", "a b"), "wss://h/ws?clientId=a%20b");
  assert.deepEqual(outputFiles({ images: [{ filename: "output_00001_.mp4", subfolder: "", type: "output" }], animated: [true] }), [
    { filename: "output_00001_.mp4", subfolder: "", type: "output" },
  ]);
  assert.deepEqual(outputFiles({ gifs: [{ filename: "a.mp4" }] }), [{ filename: "a.mp4", subfolder: "", type: "output" }]);
});

test("custom fields: graph default, replaced only with the same type", () => {
  const g = parseWorkflow(ltx);
  const params = [
    { id: "a", label: "Seed", node: "5516:4832", field: "noise_seed" },
    { id: "b", label: "LoRA", node: "5004:5606", field: "strength_model" },
    { id: "c", label: "Prompt", node: "5508", field: "value" },
    { id: "d", label: "Gone", node: "none", field: "x" },
  ];
  assert.equal(paramDefault(g, params[0]), 42);
  const out = withParams(g, params, { a: 7, b: "1.2", c: "other prompt", d: 3 });
  assert.equal(out["5516:4832"].inputs.noise_seed, 7);
  assert.equal(out["5004:5606"].inputs.strength_model, 1.5, "text in a numeric field is ignored");
  assert.equal(out["5508"].inputs.value, "other prompt");
  assert.equal(g["5516:4832"].inputs.noise_seed, 42, "the original is unchanged");
  const ksampler = listNodes(g).find((n) => n.id === "5014:5506")!;
  assert.deepEqual(editableFields(ksampler), ["value"]);
});
