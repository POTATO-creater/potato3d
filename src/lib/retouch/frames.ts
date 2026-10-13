// Quais frames restaurados entram no ajuste. Frames vizinhos de um clipe são
// quase iguais, e restaurados demais em relação às fotos puxam o splat para os
// detalhes que o modelo de vídeo reinventou: o centro, que as fotos veem bem,
// perde nitidez (scripts/restore-test: 1 a cada 3 frames recuperou 2 dB nas
// fotos e manteve o fantasma removido).

/** Padrão: frames restaurados até 25% do número de fotos. */
export const DEFAULT_RESTORED_RATIO = 0.25;

/** Total de frames restaurados permitido para `photos` fotos. */
export const restoredLimit = (photos: number, ratio: number) => Math.max(1, Math.round(photos * ratio));

/** Índices escolhidos em cada clipe (`counts` = frames de cada um), no máximo
 *  `max` no total. A cota de cada clipe é proporcional ao tamanho dele (pelo
 *  menos 1), e os frames são espaçados por igual no meio de cada trecho, sem
 *  pegar as pontas (quase na pose de uma foto real). */
export function pickRestoredFrames(counts: number[], max: number): number[][] {
  const total = counts.reduce((a, b) => a + b, 0);
  if (max >= total) return counts.map((n) => Array.from({ length: n }, (_, i) => i));
  const quota = counts.map((n) => (n ? Math.max(1, Math.round((max * n) / total)) : 0));
  // Ajuste fino para a soma bater com `max` (tirando dos clipes maiores).
  let excess = quota.reduce((a, b) => a + b, 0) - Math.max(max, counts.filter(Boolean).length);
  while (excess > 0) {
    const i = quota.indexOf(Math.max(...quota));
    if (quota[i] <= 1) break;
    quota[i]--;
    excess--;
  }
  return counts.map((n, c) => {
    const k = Math.min(quota[c], n);
    return Array.from({ length: k }, (_, j) => Math.min(n - 1, Math.floor(((j + 0.5) * n) / k)));
  });
}
