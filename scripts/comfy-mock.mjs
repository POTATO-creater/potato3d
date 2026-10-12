#!/usr/bin/env node
// ComfyUI de mentira para desenvolver e testar a integração sem GPU: as rotas
// que o app usa (/system_stats, /upload/image, /prompt, /history, /view,
// /queue, /interrupt), o WebSocket /ws com o progresso de uma execução e os
// cabeçalhos CORS do --enable-cors-header.
//
//   node scripts/comfy-mock.mjs [--port 8188] [--output video.mp4] [--step-ms 120] [--fail]
//
// Sem --output, devolve o próprio vídeo enviado (restauração "identidade").
// --fail faz a execução terminar com execution_error no amostrador.

import { createHash, randomUUID } from "node:crypto";
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { basename } from "node:path";

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const port = Number(opt("port", 8188));
const stepMs = Number(opt("step-ms", 120));
const outputPath = opt("output", null);
const fail = args.includes("--fail");

const uploads = new Map(); // nome → Buffer
const outputs = new Map(); // nome → Buffer
const history = new Map(); // prompt_id → entrada do /history
const sockets = new Map(); // clientId → socket
let cancelled = new Set();

const cors = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function send(res, status, body, type = "application/json") {
  res.writeHead(status, { ...cors, "Content-Type": type });
  res.end(type === "application/json" ? JSON.stringify(body) : body);
}

const readBody = (req) =>
  new Promise((resolve) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
  });

/** Multipart mínimo: campos de texto e o arquivo "image". */
function parseMultipart(buf, contentType) {
  const boundary = Buffer.from(`--${/boundary=(.+)$/.exec(contentType)[1]}`);
  const fields = {};
  let file = null;
  let pos = buf.indexOf(boundary);
  while (pos >= 0) {
    const next = buf.indexOf(boundary, pos + boundary.length);
    if (next < 0) break;
    const part = buf.subarray(pos + boundary.length + 2, next - 2);
    const sep = part.indexOf("\r\n\r\n");
    const head = part.subarray(0, sep).toString();
    const body = part.subarray(sep + 4);
    const name = /name="([^"]+)"/.exec(head)?.[1];
    const filename = /filename="([^"]*)"/.exec(head)?.[1];
    if (filename != null) file = { name: basename(filename), data: body };
    else if (name) fields[name] = body.toString();
    pos = next;
  }
  return { fields, file };
}

// WebSocket: só o necessário para mandar texto ao cliente.
function wsFrame(text) {
  const payload = Buffer.from(text);
  const len = payload.length;
  const head = len < 126 ? Buffer.from([0x81, len]) : len < 65536 ? Buffer.from([0x81, 126, len >> 8, len & 255]) : null;
  if (!head) throw new Error("mensagem grande demais para o mock");
  return Buffer.concat([head, payload]);
}
const emit = (clientId, type, data) => {
  const s = sockets.get(clientId);
  if (s && !s.destroyed) s.write(wsFrame(JSON.stringify({ type, data })));
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function execute(promptId, clientId, prompt) {
  const ids = Object.keys(prompt);
  const saveNode = ids.find((id) => /^Save|Combine/.test(prompt[id].class_type)) ?? ids[ids.length - 1];
  const loadNode = ids.find((id) => /^(VHS_)?LoadVideo/.test(prompt[id].class_type));
  const inputName = loadNode ? prompt[loadNode].inputs.file ?? prompt[loadNode].inputs.video : null;
  const sampler = ids.find((id) => /Sampler/.test(prompt[id].class_type));
  emit(clientId, "status", { status: { exec_info: { queue_remaining: 1 } } });
  await sleep(stepMs);
  emit(clientId, "execution_start", { prompt_id: promptId });
  const cached = ids.filter((id) => /Loader/.test(prompt[id].class_type));
  emit(clientId, "execution_cached", { prompt_id: promptId, nodes: cached });
  for (const id of ids.filter((x) => !cached.includes(x))) {
    if (cancelled.has(promptId)) {
      emit(clientId, "execution_interrupted", { prompt_id: promptId, node_id: id });
      history.set(promptId, { outputs: {}, status: { status_str: "error", completed: false, messages: [] } });
      return;
    }
    emit(clientId, "executing", { prompt_id: promptId, node: id, display_node: id });
    if (id === sampler) {
      for (let v = 1; v <= 8; v++) {
        await sleep(stepMs * 2);
        emit(clientId, "progress", { prompt_id: promptId, node: id, value: v, max: 8 });
        if (fail && v === 4) {
          const err = { prompt_id: promptId, node_id: id, node_type: prompt[id].class_type, exception_message: "CUDA out of memory (mock)" };
          emit(clientId, "execution_error", err);
          history.set(promptId, { outputs: {}, status: { status_str: "error", completed: false, messages: [["execution_error", err]] } });
          return;
        }
      }
    } else await sleep(stepMs / 4);
  }
  const data = outputPath ? readFileSync(outputPath) : uploads.get(inputName);
  if (!data) {
    const err = { prompt_id: promptId, node_id: loadNode, node_type: "LoadVideo", exception_message: `arquivo de entrada não enviado: ${inputName}` };
    emit(clientId, "execution_error", err);
    history.set(promptId, { outputs: {}, status: { status_str: "error", completed: false, messages: [["execution_error", err]] } });
    return;
  }
  const filename = `output_${String(outputs.size + 1).padStart(5, "0")}_.mp4`;
  outputs.set(filename, data);
  const output = { images: [{ filename, subfolder: "", type: "output" }], animated: [true] };
  emit(clientId, "executed", { prompt_id: promptId, node: saveNode, output });
  history.set(promptId, { prompt: [0, promptId, prompt], outputs: { [saveNode]: output }, status: { status_str: "success", completed: true, messages: [] } });
  emit(clientId, "execution_success", { prompt_id: promptId });
  emit(clientId, "executing", { prompt_id: promptId, node: null });
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  if (req.method === "OPTIONS") return send(res, 204, "", "text/plain");
  const route = `${req.method} ${url.pathname}`;
  if (route === "GET /system_stats")
    return send(res, 200, { system: { comfyui_version: "0.0.0-mock", os: "mock" }, devices: [{ name: "cuda:0 Mock GPU", type: "cuda", vram_total: 24 * 1024 ** 3 }] });
  if (route === "POST /upload/image") {
    const { file } = parseMultipart(await readBody(req), req.headers["content-type"]);
    if (!file) return send(res, 400, { error: "sem arquivo" });
    uploads.set(file.name, file.data);
    console.log(`upload ${file.name} (${(file.data.length / 1e6).toFixed(1)} MB)`);
    return send(res, 200, { name: file.name, subfolder: "", type: "input" });
  }
  if (route === "POST /prompt") {
    const { prompt, client_id } = JSON.parse((await readBody(req)).toString());
    if (!prompt || typeof prompt !== "object") return send(res, 400, { error: { message: "prompt inválido" }, node_errors: {} });
    const promptId = randomUUID();
    console.log(`prompt ${promptId} (${Object.keys(prompt).length} nós) de ${client_id}`);
    send(res, 200, { prompt_id: promptId, number: 1, node_errors: {} });
    void execute(promptId, client_id, prompt);
    return;
  }
  if (req.method === "GET" && url.pathname.startsWith("/history/")) {
    const id = decodeURIComponent(url.pathname.slice("/history/".length));
    return send(res, 200, history.has(id) ? { [id]: history.get(id) } : {});
  }
  if (route === "GET /view") {
    const data = outputs.get(url.searchParams.get("filename"));
    return data ? send(res, 200, data, "video/mp4") : send(res, 404, { error: "não encontrado" });
  }
  if (route === "POST /queue" || route === "POST /interrupt") {
    const body = JSON.parse((await readBody(req)).toString() || "{}");
    for (const id of [...(body.delete ?? []), body.prompt_id].filter(Boolean)) cancelled.add(id);
    return send(res, 200, {});
  }
  send(res, 404, { error: `rota desconhecida: ${route}` });
});

server.on("upgrade", (req, socket) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  if (url.pathname !== "/ws") return socket.destroy();
  const accept = createHash("sha1").update(`${req.headers["sec-websocket-key"]}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`).digest("base64");
  socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
  const clientId = url.searchParams.get("clientId") ?? randomUUID();
  sockets.set(clientId, socket);
  socket.on("data", (buf) => {
    if ((buf[0] & 0x0f) === 0x8) socket.end(); // close do cliente
  });
  socket.on("close", () => sockets.delete(clientId));
  socket.write(wsFrame(JSON.stringify({ type: "status", data: { status: { exec_info: { queue_remaining: 0 } }, sid: clientId } })));
});

server.listen(port, "127.0.0.1", () =>
  console.log(`ComfyUI simulado em http://127.0.0.1:${port}${outputPath ? ` · saída: ${outputPath}` : " · saída = entrada"}${fail ? " · falha no amostrador" : ""}`),
);
