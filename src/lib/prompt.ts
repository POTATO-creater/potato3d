import type { SubjectPrompt } from "./api";

const EPS = 0.05;
/** Raio (fração da largura) em que o Shift+clique acha um ponto. */
const HIT = 0.04;

export type PromptMarks = Pick<SubjectPrompt, "box" | "points">;

/** A marcação ainda vale? Fora do trecho, o frame mais próximo dela seria
 *  outro momento do vídeo (a caixa cairia em cima de outra imagem). */
export function promptInTrim(p: SubjectPrompt | null | undefined, trim: { start: number; end: number | null }): boolean {
  if (!p || p.time == null) return true;
  return p.time >= trim.start - EPS && (trim.end == null || p.time <= trim.end + EPS);
}

/** Marcas sobre a base (instante ou imagem); sem caixa nem pontos, volta ao automático. */
export function withMarks(base: SubjectPrompt, m: PromptMarks): SubjectPrompt | null {
  return m.box || m.points.length ? { ...base, box: m.box, points: m.points } : null;
}

/** Marcas depois de um Shift+clique em (x, y): some o ponto mais perto, ou a caixa. */
export function eraseAt(m: PromptMarks, x: number, y: number, aspect: number): PromptMarks {
  let best = -1, bestD = HIT;
  m.points.forEach((p, i) => {
    // distância em proporção de largura (y escalado pelo formato do quadro)
    const d = Math.hypot(p.x - x, (p.y - y) / aspect);
    if (d <= bestD) [best, bestD] = [i, d];
  });
  if (best >= 0) return { ...m, points: m.points.filter((_, i) => i !== best) };
  const b = m.box;
  if (b && x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) return { ...m, box: null };
  return m;
}
