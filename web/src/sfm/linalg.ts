// Álgebra linear densa e pequena para o SfM: matrizes como Float64Array
// linha a linha, 3×3 como `Mat3` (9 números).

export type Vec3 = [number, number, number];
export type Mat3 = Float64Array; // 9, linha a linha

export const mat3 = (...v: number[]): Mat3 => Float64Array.from(v);
export const eye3 = (): Mat3 => mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);

export function mul3(a: Mat3, b: Mat3): Mat3 {
  const r = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  return r;
}

export function transpose3(a: Mat3): Mat3 {
  return mat3(a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]);
}

export function mulVec3(a: Mat3, v: ArrayLike<number>): Vec3 {
  return [
    a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
    a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
    a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
  ];
}

export function mulTVec3(a: Mat3, v: ArrayLike<number>): Vec3 {
  return [
    a[0] * v[0] + a[3] * v[1] + a[6] * v[2],
    a[1] * v[0] + a[4] * v[1] + a[7] * v[2],
    a[2] * v[0] + a[5] * v[1] + a[8] * v[2],
  ];
}

export const cross = (a: ArrayLike<number>, b: ArrayLike<number>): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm = (a: ArrayLike<number>) => Math.hypot(a[0], a[1], a[2]);
export const skew = (v: ArrayLike<number>): Mat3 => mat3(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0);

export function det3(a: Mat3) {
  return (
    a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6])
  );
}

/** exp([w]×): rotação a partir do vetor eixo·ângulo. */
export function expSO3(w: ArrayLike<number>): Mat3 {
  const th = Math.hypot(w[0], w[1], w[2]);
  const K = skew(w);
  const K2 = mul3(K, K);
  const a = th < 1e-10 ? 1 : Math.sin(th) / th;
  const b = th < 1e-10 ? 0.5 : (1 - Math.cos(th)) / (th * th);
  const R = eye3();
  for (let i = 0; i < 9; i++) R[i] += a * K[i] + b * K2[i];
  return R;
}

/** Quatérnio (w, x, y, z) de uma rotação, como o COLMAP grava. */
export function quatFromMat(R: Mat3): [number, number, number, number] {
  const tr = R[0] + R[4] + R[8];
  let w, x, y, z;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s;
    x = (R[7] - R[5]) / s;
    y = (R[2] - R[6]) / s;
    z = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
    w = (R[7] - R[5]) / s;
    x = 0.25 * s;
    y = (R[1] + R[3]) / s;
    z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
    w = (R[2] - R[6]) / s;
    x = (R[1] + R[3]) / s;
    y = 0.25 * s;
    z = (R[5] + R[7]) / s;
  } else {
    const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
    w = (R[3] - R[1]) / s;
    x = (R[2] + R[6]) / s;
    y = (R[5] + R[7]) / s;
    z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z);
  return w < 0 ? [-w / n, -x / n, -y / n, -z / n] : [w / n, x / n, y / n, z / n];
}

/** Rotação mais próxima de M (decomposição polar via SVD). */
export function nearestRotation(M: Mat3): Mat3 {
  const { U, V } = svd3(M);
  let R = mul3(U, transpose3(V));
  if (det3(R) < 0) {
    for (let i = 0; i < 3; i++) U[i * 3 + 2] = -U[i * 3 + 2];
    R = mul3(U, transpose3(V));
  }
  return R;
}

/**
 * Autovalores/autovetores de uma matriz simétrica n×n (Jacobi cíclico).
 * Retorna autovalores em ordem crescente e os vetores como colunas de `V`.
 */
export function symEig(A: Float64Array, n: number): { values: Float64Array; V: Float64Array } {
  const a = Float64Array.from(A);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = [...Array(n).keys()].sort((i, j) => a[i * n + i] - a[j * n + j]);
  const values = new Float64Array(n);
  const Vs = new Float64Array(n * n);
  order.forEach((src, dst) => {
    values[dst] = a[src * n + src];
    for (let k = 0; k < n; k++) Vs[k * n + dst] = V[k * n + src];
  });
  return { values, V: Vs };
}

/** Autovetor do menor autovalor de AᵀA (A com `cols` colunas, linha a linha). */
export function nullVector(A: ArrayLike<number>, rows: number, cols: number): Float64Array {
  const AtA = new Float64Array(cols * cols);
  for (let r = 0; r < rows; r++)
    for (let i = 0; i < cols; i++) {
      const ai = A[r * cols + i];
      if (ai === 0) continue;
      for (let j = i; j < cols; j++) AtA[i * cols + j] += ai * A[r * cols + j];
    }
  for (let i = 0; i < cols; i++) for (let j = 0; j < i; j++) AtA[i * cols + j] = AtA[j * cols + i];
  const { V } = symEig(AtA, cols);
  const v = new Float64Array(cols);
  for (let k = 0; k < cols; k++) v[k] = V[k * cols];
  return v;
}

/** SVD 3×3: M = U diag(S) Vᵀ, S decrescente. */
export function svd3(M: Mat3): { U: Mat3; S: Vec3; V: Mat3 } {
  const MtM = mul3(transpose3(M), M);
  const { values, V: Ve } = symEig(MtM, 3);
  // ordem decrescente
  const V = mat3(Ve[2], Ve[1], Ve[0], Ve[5], Ve[4], Ve[3], Ve[8], Ve[7], Ve[6]);
  const S: Vec3 = [Math.sqrt(Math.max(values[2], 0)), Math.sqrt(Math.max(values[1], 0)), Math.sqrt(Math.max(values[0], 0))];
  const cols: Vec3[] = [];
  for (let k = 0; k < 2; k++) {
    const u = mulVec3(M, [V[k], V[3 + k], V[6 + k]]);
    const n = norm(u) || 1;
    cols.push([u[0] / n, u[1] / n, u[2] / n]);
  }
  cols.push(cross(cols[0], cols[1]));
  const U = mat3(cols[0][0], cols[1][0], cols[2][0], cols[0][1], cols[1][1], cols[2][1], cols[0][2], cols[1][2], cols[2][2]);
  return { U, S, V };
}

/** Resolve A x = b com A simétrica definida positiva (Cholesky, in-place em cópia). */
export function choleskySolve(A: Float64Array, b: Float64Array, n: number): Float64Array | null {
  const L = Float64Array.from(A);
  for (let j = 0; j < n; j++) {
    let d = L[j * n + j];
    for (let k = 0; k < j; k++) d -= L[j * n + k] ** 2;
    if (!(d > 0)) return null;
    d = Math.sqrt(d);
    L[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let s = L[i * n + j];
      const Li = i * n, Lj = j * n;
      for (let k = 0; k < j; k++) s -= L[Li + k] * L[Lj + k];
      L[i * n + j] = s / d;
    }
  }
  const y = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    let s = y[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * y[k];
    y[i] = s / L[i * n + i];
  }
  return y;
}

/** Inversa de uma 3×3 simétrica (blocos de ponto no BA). */
export function inv3(a: ArrayLike<number>): Mat3 | null {
  const d = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (!(Math.abs(d) > 1e-300)) return null;
  const id = 1 / d;
  return mat3(
    (a[4] * a[8] - a[5] * a[7]) * id,
    (a[2] * a[7] - a[1] * a[8]) * id,
    (a[1] * a[5] - a[2] * a[4]) * id,
    (a[5] * a[6] - a[3] * a[8]) * id,
    (a[0] * a[8] - a[2] * a[6]) * id,
    (a[2] * a[3] - a[0] * a[5]) * id,
    (a[3] * a[7] - a[4] * a[6]) * id,
    (a[1] * a[6] - a[0] * a[7]) * id,
    (a[0] * a[4] - a[1] * a[3]) * id,
  );
}
