import type { SubjectPrompt } from "./api";

const EPS = 0.05;
/** Radius (fraction of width) within which Shift+click hits a point. */
const HIT = 0.04;

export type PromptMarks = Pick<SubjectPrompt, "box" | "points">;

/** Is the marking still valid? Outside the trim range, the nearest frame
 *  would be another moment of the video (the box would land on another image). */
export function promptInTrim(p: SubjectPrompt | null | undefined, trim: { start: number; end: number | null }): boolean {
  if (!p || p.time == null) return true;
  return p.time >= trim.start - EPS && (trim.end == null || p.time <= trim.end + EPS);
}

/** Marks on top of the base (time or image); with no box or points, falls back to automatic. */
export function withMarks(base: SubjectPrompt, m: PromptMarks): SubjectPrompt | null {
  return m.box || m.points.length ? { ...base, box: m.box, points: m.points } : null;
}

/** Marks after a Shift+click at (x, y): removes the nearest point, or the box. */
export function eraseAt(m: PromptMarks, x: number, y: number, aspect: number): PromptMarks {
  let best = -1, bestD = HIT;
  m.points.forEach((p, i) => {
    // distance as a fraction of width (y scaled by the frame aspect)
    const d = Math.hypot(p.x - x, (p.y - y) / aspect);
    if (d <= bestD) [best, bestD] = [i, d];
  });
  if (best >= 0) return { ...m, points: m.points.filter((_, i) => i !== best) };
  const b = m.box;
  if (b && x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) return { ...m, box: null };
  return m;
}
