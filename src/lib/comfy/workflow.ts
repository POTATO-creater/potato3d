// ComfyUI workflows in API format (the editor's "Export (API)"): an object
// { "<node id>": { class_type, inputs, _meta?: { title } } }. This is the
// format POST /prompt accepts. The editor format (nodes/links) won't do: it
// lacks the resolved widget values.

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
  /** Inputs with a literal value (widgets), not links to other nodes. */
  literalInputs: { name: string; value: unknown }[];
}

/** Where the app injects the uploaded file and where it reads the result. */
export interface WorkflowIO {
  input: { node: string; field: string };
  output: { node: string };
}

export type WorkflowError = "notJson" | "editorFormat" | "notWorkflow" | "empty";

export class WorkflowParseError extends Error {
  readonly reason: WorkflowError;
  constructor(reason: WorkflowError) {
    super(reason);
    this.reason = reason;
  }
}

/** A link in API format is `[source node id, output index]`. */
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

/** File field of the most common video/image loader nodes. */
const INPUT_FIELDS: Record<string, string> = {
  LoadVideo: "file",
  VHS_LoadVideo: "video",
  VHS_LoadVideoPath: "video",
  LoadImage: "image",
};

const OUTPUT_CLASSES = ["SaveVideo", "VHS_VideoCombine", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveWEBM", "SaveImage"];

/** String fields of a node that can take a file name. */
export function fileFields(node: NodeInfo): string[] {
  const known = INPUT_FIELDS[node.classType];
  const strings = node.literalInputs.filter((i) => typeof i.value === "string").map((i) => i.name);
  return known && strings.includes(known) ? [known, ...strings.filter((s) => s !== known)] : strings;
}

/** Initial guess for input and output; the user can change it in settings. */
export function guessIO(graph: ApiGraph): WorkflowIO | null {
  const nodes = listNodes(graph);
  const input =
    nodes.find((n) => n.classType in INPUT_FIELDS && fileFields(n)[0] === INPUT_FIELDS[n.classType]) ??
    nodes.find((n) => /load/i.test(n.classType) && fileFields(n).length > 0);
  const output = OUTPUT_CLASSES.map((c) => nodes.find((n) => n.classType === c)).find(Boolean);
  if (!input || !output) return null;
  return { input: { node: input.id, field: fileFields(input)[0] }, output: { node: output.id } };
}

export const DEFAULT_FPS = 24;

/** Reference video frame rate for a workflow: the graph's first literal
 *  `fps`/`frame_rate` (the output video uses that rate), otherwise 24. */
export function workflowFps(graph: ApiGraph): number {
  for (const n of listNodes(graph)) {
    const f = n.literalInputs.find((i) => (i.name === "fps" || i.name === "frame_rate") && typeof i.value === "number");
    if (f && (f.value as number) > 0) return f.value as number;
  }
  return DEFAULT_FPS;
}

/** Problems of the input/output configuration against the graph. */
export type IOProblem = "inputNode" | "inputField" | "outputNode";

export function checkIO(graph: ApiGraph, io: WorkflowIO): IOProblem[] {
  const problems: IOProblem[] = [];
  const input = graph[io.input.node];
  if (!input) problems.push("inputNode");
  else if (!(io.input.field in input.inputs) || isLink(input.inputs[io.input.field])) problems.push("inputField");
  if (!graph[io.output.node]) problems.push("outputNode");
  return problems;
}

/** Copy of the graph with the uploaded file in the input field. The workflow
 *  takes everything else (duration, resolution) from the video itself. */
export function withInput(graph: ApiGraph, io: WorkflowIO, filename: string): ApiGraph {
  const copy = structuredClone(graph);
  copy[io.input.node].inputs[io.input.field] = filename;
  return copy;
}

/** A node field exposed in the app for the user to edit before sending. */
export interface WorkflowParam {
  id: string;
  label: string;
  node: string;
  field: string;
}

export type ParamValue = string | number | boolean;

/** Current field value in the graph (the default), if it is a simple value. */
export function paramDefault(graph: ApiGraph, p: Pick<WorkflowParam, "node" | "field">): ParamValue | undefined {
  const v = graph[p.node]?.inputs[p.field];
  return typeof v === "string" || typeof v === "number" || typeof v === "boolean" ? v : undefined;
}

/** A node's fields with simple values (text, number, on/off). */
export function editableFields(node: NodeInfo): string[] {
  return node.literalInputs.filter((i) => ["string", "number", "boolean"].includes(typeof i.value)).map((i) => i.name);
}

/** Copy of the graph with the custom field values. Values whose type differs
 *  from the original are ignored (the graph changed since they were chosen). */
export function withParams(graph: ApiGraph, params: WorkflowParam[], values: Record<string, ParamValue | undefined>): ApiGraph {
  const copy = structuredClone(graph);
  for (const p of params) {
    const v = values[p.id];
    const def = paramDefault(copy, p);
    if (v === undefined || def === undefined || typeof v !== typeof def) continue;
    copy[p.node].inputs[p.field] = v;
  }
  return copy;
}
