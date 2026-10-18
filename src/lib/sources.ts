// Picked or dropped paths → batch inputs: each video, package or frames folder
// becomes its own item; loose images are grouped by the folder they are in.

export type SourceInput =
  | { kind: "package"; path: string }
  | { kind: "video"; path: string }
  | { kind: "images"; paths: string[] };

export const VIDEO_EXT = ["mp4", "mov", "m4v", "webm", "mkv", "avi"];

const extOf = (p: string) => {
  const name = p.split(/[\\/]/).pop() ?? p;
  return name.includes(".") ? (name.split(".").pop()?.toLowerCase() ?? "") : "";
};
const dirOf = (p: string) => p.replace(/[\\/][^\\/]*$/, "");

export function groupSources(paths: string[], imageExt: string[], packageExt: string): SourceInput[] {
  const out: SourceInput[] = [];
  const byDir = new Map<string, string[]>();
  for (const p of paths) {
    const ext = extOf(p);
    if (ext === packageExt) out.push({ kind: "package", path: p });
    else if (VIDEO_EXT.includes(ext)) out.push({ kind: "video", path: p });
    else if (imageExt.includes(ext)) {
      const dir = dirOf(p);
      let group = byDir.get(dir);
      if (!group) {
        byDir.set(dir, (group = []));
        out.push({ kind: "images", paths: group });
      }
      group.push(p);
    }
    // Anything else: a folder (its name may have a dot), expanded by
    // `probeImages`, which fails if it holds no images.
    else out.push({ kind: "images", paths: [p] });
  }
  return out;
}
