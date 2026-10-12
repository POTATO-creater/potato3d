// Workflows do ComfyUI no formato de API (o "Export (API)" do editor): um
// objeto { "<id do nó>": { class_type, inputs, _meta?: { title } } }. É o
// formato que o POST /prompt aceita. O formato do editor (nodes/links) não
// serve: faltam os valores resolvidos dos widgets.

export interface ApiNode {
  class_type: string;
  inputs: Record<string, unknown>;
  _meta?: { title?: string };
}

export type ApiGraph = Record<string, ApiNode>;

export interface NodeInfo {
  id: string;
  classType: string;
  title: string;
  /** Entradas com valor literal (widgets), não ligações a outros nós. */
  literalInputs: { name: string; value: unknown }[];
}

/** Campo numérico que recebe a duração do vídeo enviado. */
export interface LengthParam {
  node: string;
  field: string;
  unit: "seconds" | "frames";
}

/** Onde o app injeta o arquivo enviado e onde lê o resultado. */
export interface WorkflowIO {
  input: { node: string; field: string };
  output: { node: string };
  /** Duração do vídeo. `undefined`: ainda não escolhido (vale o palpite);
   *  `null`: o workflow não tem esse campo. */
  length?: LengthParam | null;
  /** Taxa de quadros do vídeo de referência; `undefined` vale o palpite. */
  fps?: number;
}

export type WorkflowError = "notJson" | "editorFormat" | "notWorkflow" | "empty";

export class WorkflowParseError extends Error {
  readonly reason: WorkflowError;
  constructor(reason: WorkflowError) {
    super(reason);
    this.reason = reason;
  }
}

/** Uma ligação no formato de API é `[id do nó de origem, índice da saída]`. */
const isLink = (v: unknown) => Array.isArray(v) && v.length === 2 && typeof v[0] === "string" && typeof v[1] === "number";

export function parseWorkflow(text: string): ApiGraph {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new WorkflowParseError("notJson");
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) throw new WorkflowParseError("notWorkflow");
  const obj = data as Record<string, unknown>;
  if (Array.isArray(obj.nodes) && Array.isArray(obj.links)) throw new WorkflowParseError("editorFormat");
  const entries = Object.entries(obj);
  if (!entries.length) throw new WorkflowParseError("empty");
  for (const [, node] of entries) {
    const n = node as Partial<ApiNode> | null;
    if (!n || typeof n !== "object" || typeof n.class_type !== "string" || !n.inputs || typeof n.inputs !== "object") {
      throw new WorkflowParseError("notWorkflow");
    }
  }
  return obj as ApiGraph;
}

export function listNodes(graph: ApiGraph): NodeInfo[] {
  return Object.entries(graph)
    .map(([id, n]) => ({
      id,
      classType: n.class_type,
      title: n._meta?.title || n.class_type,
      literalInputs: Object.entries(n.inputs)
        .filter(([, v]) => !isLink(v))
        .map(([name, value]) => ({ name, value })),
    }))
    .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
}

/** Campo de arquivo dos nós de carregar vídeo/imagem mais comuns. */
const INPUT_FIELDS: Record<string, string> = {
  LoadVideo: "file",
  VHS_LoadVideo: "video",
  VHS_LoadVideoPath: "video",
  LoadImage: "image",
};

const OUTPUT_CLASSES = ["SaveVideo", "VHS_VideoCombine", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveWEBM", "SaveImage"];

/** Campos string de um nó que podem receber um nome de arquivo. */
export function fileFields(node: NodeInfo): string[] {
  const known = INPUT_FIELDS[node.classType];
  const strings = node.literalInputs.filter((i) => typeof i.value === "string").map((i) => i.name);
  return known && strings.includes(known) ? [known, ...strings.filter((s) => s !== known)] : strings;
}

/** Palpite inicial de entrada e saída; o usuário pode trocar na configuração. */
export function guessIO(graph: ApiGraph): WorkflowIO | null {
  const nodes = listNodes(graph);
  const input =
    nodes.find((n) => n.classType in INPUT_FIELDS && fileFields(n)[0] === INPUT_FIELDS[n.classType]) ??
    nodes.find((n) => /load/i.test(n.classType) && fileFields(n).length > 0);
  const output = OUTPUT_CLASSES.map((c) => nodes.find((n) => n.classType === c)).find(Boolean);
  if (!input || !output) return null;
  return { input: { node: input.id, field: fileFields(input)[0] }, output: { node: output.id } };
}

/** Campos numéricos (widgets) de um nó. */
export const numberFields = (node: NodeInfo) => node.literalInputs.filter((i) => typeof i.value === "number").map((i) => i.name);

/** Palpite do campo de duração: `duration`/`seconds` em segundos;
 *  `length`/`num_frames`/`frame_count` de nós de vídeo ou latente, em frames. */
export function guessLength(graph: ApiGraph): LengthParam | null {
  for (const n of listNodes(graph)) {
    const fields = numberFields(n);
    const sec = fields.find((f) => f === "duration" || f === "seconds");
    if (sec) return { node: n.id, field: sec, unit: "seconds" };
    const fr = fields.find((f) => f === "num_frames" || f === "frame_count" || (f === "length" && /video|latent/i.test(n.classType)));
    if (fr) return { node: n.id, field: fr, unit: "frames" };
  }
  return null;
}

/** Palpite da taxa de quadros: o primeiro `fps`/`frame_rate` literal do grafo. */
export function guessFps(graph: ApiGraph): number | null {
  for (const n of listNodes(graph)) {
    const f = n.literalInputs.find((i) => (i.name === "fps" || i.name === "frame_rate") && typeof i.value === "number");
    if (f && (f.value as number) > 0) return f.value as number;
  }
  return null;
}

export const DEFAULT_FPS = 24;

/** Configuração em vigor, com os palpites no lugar do que não foi escolhido. */
export function effectiveIO(graph: ApiGraph, io: WorkflowIO): Required<WorkflowIO> {
  return {
    ...io,
    length: io.length === undefined ? guessLength(graph) : io.length,
    fps: io.fps ?? guessFps(graph) ?? DEFAULT_FPS,
  };
}

/** Problemas da configuração de entrada/saída diante do grafo. */
export type IOProblem = "inputNode" | "inputField" | "outputNode" | "lengthField";

export function checkIO(graph: ApiGraph, io: WorkflowIO): IOProblem[] {
  const problems: IOProblem[] = [];
  const input = graph[io.input.node];
  if (!input) problems.push("inputNode");
  else if (!(io.input.field in input.inputs) || isLink(input.inputs[io.input.field])) problems.push("inputField");
  if (!graph[io.output.node]) problems.push("outputNode");
  const len = io.length;
  if (len && typeof graph[len.node]?.inputs[len.field] !== "number") problems.push("lengthField");
  return problems;
}

/** Cópia do grafo com o arquivo enviado no campo de entrada e, se houver
 *  campo de duração, a duração do vídeo. */
export function withInput(graph: ApiGraph, io: WorkflowIO, filename: string, video?: { frames: number; fps: number }): ApiGraph {
  const copy = structuredClone(graph);
  copy[io.input.node].inputs[io.input.field] = filename;
  const len = io.length;
  if (video && len && copy[len.node]) {
    copy[len.node].inputs[len.field] = len.unit === "frames" ? video.frames : Math.round((video.frames / video.fps) * 100) / 100;
  }
  return copy;
}
