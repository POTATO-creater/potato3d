// Trajetórias de retoque, em convenção COLMAP (pose = mundo → câmera, câmera
// olhando para +Z com +Y para baixo). Saem de uma foto real, passam por
// ângulos que a órbita não viu e voltam para uma foto real: as pontas
// (âncoras) têm exatamente a pose de uma foto, o que dá ao modelo de vídeo
// uma referência real e permite medir a restauração. Porte do
// build_trajectory de scripts/restore-test/restore_test.py.

export type Vec3 = [number, number, number];
/** (w, x, y, z), a ordem do COLMAP. */
export type Quat = [number, number, number, number];
type Mat3 = [Vec3, Vec3, Vec3];

export interface CameraPose {
  name: string;
  qvec: Quat;
  tvec: Vec3;
}

export interface Lens {
  width: number;
  height: number;
  fx: number;
  fy: number;
}

export interface PathPose {
  qvec: Quat;
  tvec: Vec3;
  /** Nome da foto quando a pose é exatamente a dela. */
  anchor: string | null;
}

export interface ClipPlan {
  size: [number, number];
  intrinsics: { fx: number; fy: number; cx: number; cy: number };
  poses: PathPose[];
}

export interface ClipOptions {
  /** Frames do clipe; o LTX quer 8n + 1. */
  frames: number;
  /** Lado maior do vídeo, arredondado (com o menor) para múltiplos de 32. */
  longSide: number;
  /** Fração do clipe parada em cada ponta (âncoras). */
  hold: number;
  /** Fração do clipe em cada transição. */
  ramp: number;
  /** Proporção largura/altura; sem ela, a da lente. */
  aspect?: number;
}

/** Sem pontas paradas: a transição usa 40% do clipe de cada lado, o que deixa
 *  a câmera bem mais lenta que com pontas paradas (o LTX borra o que passa
 *  rápido). */
export const DEFAULT_CLIP: ClipOptions = { frames: 97, longSide: 960, hold: 0, ramp: 0.4 };

/** Frames de um clipe de `seconds` a `fps`, no 8n + 1 mais próximo (o LTX
 *  trabalha em blocos de 8 frames mais o primeiro). Mínimo de 17. */
export function clipFrames(seconds: number, fps: number): number {
  const n = Math.round((seconds * fps - 1) / 8);
  return Math.max(2, n) * 8 + 1;
}

// ---------------------------------------------------------------- álgebra --

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: Vec3): Vec3 => scale(a, 1 / (norm(a) || 1));
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function quatToMat([w, x, y, z]: Quat): Mat3 {
  const n = Math.hypot(w, x, y, z) || 1;
  [w, x, y, z] = [w / n, x / n, y / n, z / n];
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

export function matToQuat(m: Mat3): Quat {
  const tr = m[0][0] + m[1][1] + m[2][2];
  let q: Quat;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s];
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2;
    q = [(m[2][1] - m[1][2]) / s, s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s];
  } else if (m[1][1] > m[2][2]) {
    const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2;
    q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s];
  } else {
    const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2;
    q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4];
  }
  const n = Math.hypot(...q);
  q = q.map((v) => v / n) as Quat;
  return q[0] < 0 ? (q.map((v) => -v) as Quat) : q;
}

const transpose = (m: Mat3): Mat3 => [
  [m[0][0], m[1][0], m[2][0]],
  [m[0][1], m[1][1], m[2][1]],
  [m[0][2], m[1][2], m[2][2]],
];
const mulMV = (m: Mat3, v: Vec3): Vec3 => [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
const mulMM = (a: Mat3, b: Mat3): Mat3 => {
  const bt = transpose(b);
  return a.map((row) => bt.map((col) => dot(row, col))) as Mat3;
};

export function slerp(a: Quat, b: Quat, u: number): Quat {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (d < 0) {
    b = b.map((v) => -v) as Quat;
    d = -d;
  }
  if (d > 0.9995) {
    const q = a.map((v, i) => v + u * (b[i] - v)) as Quat;
    const n = Math.hypot(...q);
    return q.map((v) => v / n) as Quat;
  }
  const th = Math.acos(d);
  const s = Math.sin(th);
  return a.map((v, i) => (Math.sin((1 - u) * th) * v + Math.sin(u * th) * b[i]) / s) as Quat;
}

/** Rotação de `ang` radianos em torno de `axis` (Rodrigues). */
function axisAngle(axis: Vec3, ang: number): Mat3 {
  const [x, y, z] = unit(axis);
  const c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
  return [
    [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
  ];
}

/** Pose câmera → mundo: rotação e centro. */
const toWorld = (p: { qvec: Quat; tvec: Vec3 }) => {
  const r = quatToMat(p.qvec);
  const rt = transpose(r);
  return { rot: rt, center: scale(mulMV(rt, p.tvec), -1) };
};
const fromWorld = (rot: Mat3, center: Vec3): { qvec: Quat; tvec: Vec3 } => {
  const r = transpose(rot);
  return { qvec: matToQuat(r), tvec: scale(mulMV(r, center), -1) };
};

/** Centro e "up" da órbita. Centro: o ponto mais perto de todos os eixos
 *  ópticos (para onde as câmeras olham); up: média do -Y das câmeras. */
export function orbitFrame(cams: CameraPose[]): { center: Vec3; up: Vec3 } {
  const a = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const b: Vec3 = [0, 0, 0];
  let up: Vec3 = [0, 0, 0];
  for (const c of cams) {
    const { rot, center } = toWorld(c);
    const fwd: Vec3 = [rot[0][2], rot[1][2], rot[2][2]];
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const p = (i === j ? 1 : 0) - fwd[i] * fwd[j];
        a[i * 3 + j] += p;
        b[i] += p * center[j];
      }
    up = add(up, [-rot[0][1], -rot[1][1], -rot[2][1]]);
  }
  return { center: solve3(a, b), up: unit(up) };
}

function solve3(m: number[], b: Vec3): Vec3 {
  const [a, bb, c, d, e, f, g, h, i] = m;
  const det = a * (e * i - f * h) - bb * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) return [0, 0, 0];
  const inv = [
    e * i - f * h, c * h - bb * i, bb * f - c * e,
    f * g - d * i, a * i - c * g, c * d - a * f,
    d * h - e * g, bb * g - a * h, a * e - bb * d,
  ];
  return [0, 1, 2].map((r) => (inv[r * 3] * b[0] + inv[r * 3 + 1] * b[1] + inv[r * 3 + 2] * b[2]) / det) as Vec3;
}

// ------------------------------------------------------------ trajetórias --

/** 0 nas pontas (âncoras), 1 no meio, com rampas em cosseno. */
export function profile(u: number, hold: number, ramp: number): number {
  const d = Math.min(u, 1 - u);
  if (d <= hold) return 0;
  if (d >= hold + ramp) return 1;
  return 0.5 - 0.5 * Math.cos((Math.PI * (d - hold)) / ramp);
}

/** Tamanho do clipe (múltiplos de 32, lado maior ~longSide) e intrínsecos
 *  com o ponto principal no centro, como o render do viewer. Sem `aspect`
 *  (largura/altura), a proporção é a da lente; com ela, o campo de visão
 *  vertical da lente se mantém e a proporção só muda a largura do quadro. */
export function clipLens(lens: Lens, longSide: number, aspect?: number): Pick<ClipPlan, "size" | "intrinsics"> {
  if (aspect && aspect > 0) {
    const r32 = (v: number) => Math.max(32, Math.round(v / 32) * 32);
    const w = r32(aspect >= 1 ? longSide : longSide * aspect);
    const h = r32(aspect >= 1 ? longSide / aspect : longSide);
    const fy = (lens.fy * h) / lens.height;
    return { size: [w, h], intrinsics: { fx: fy * (lens.fx / lens.fy), fy, cx: w / 2, cy: h / 2 } };
  }
  const s = longSide / Math.max(lens.width, lens.height);
  const w = Math.max(32, Math.round((lens.width * s) / 32) * 32);
  const h = Math.max(32, Math.round((lens.height * s) / 32) * 32);
  // A escala que cobre os dois lados; a sobra é cortada igualmente nas bordas.
  const k = Math.max(w / lens.width, h / lens.height);
  return { size: [w, h], intrinsics: { fx: lens.fx * k, fy: lens.fy * k, cx: w / 2, cy: h / 2 } };
}

/** Pose interpolada ao longo das fotos (índice fracionário, órbita fechada). */
function alongOrbit(cams: CameraPose[], s: number) {
  const m = cams.length;
  const i0 = Math.floor(s);
  const f = s - i0;
  const a = cams[((i0 % m) + m) % m], b = cams[(((i0 + 1) % m) + m) % m];
  const wa = toWorld(a), wb = toWorld(b);
  const center = add(scale(wa.center, 1 - f), scale(wb.center, f));
  const rot = quatToMat(slerp(matToQuat(transpose(wa.rot)), matToQuat(transpose(wb.rot)), f));
  const anchor = Math.abs(f) < 1e-6 ? a.name : Math.abs(f - 1) < 1e-6 ? b.name : null;
  return { rot: transpose(rot), center, anchor };
}

/** Percorre `span` fotos a partir do índice `start` com velocidade angular
 *  constante em volta do centro. As fotos quase nunca são igualmente
 *  espaçadas (e numa órbita fechada a última costuma repetir a primeira):
 *  andar "uma foto por frame" faz a câmera acelerar, frear e até parar. */
function orbitWalk(cams: CameraPose[], orbitC: Vec3, up: Vec3, start: number, span: number) {
  const m = cams.length;
  const ref: Vec3 = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = unit(cross(up, ref));
  const e2 = cross(up, e1);
  const k0 = Math.floor(start);
  const k1 = Math.ceil(start + span);
  // Azimute desembrulhado em cada foto do trecho.
  const az: number[] = [];
  for (let k = k0; k <= k1; k++) {
    const r = sub(toWorld(cams[((k % m) + m) % m]).center, orbitC);
    let a = Math.atan2(dot(r, e2), dot(r, e1));
    if (az.length) a -= 2 * Math.PI * Math.round((a - az[az.length - 1]) / (2 * Math.PI));
    az.push(a);
  }
  // Monótono no sentido da volta (ruído das poses não pode fazer voltar).
  const dir = Math.sign(az[az.length - 1] - az[0]) || 1;
  for (let i = 1; i < az.length; i++) az[i] = dir > 0 ? Math.max(az[i], az[i - 1]) : Math.min(az[i], az[i - 1]);
  const at = (s: number) => {
    const i = Math.min(Math.max(Math.floor(s) - k0, 0), az.length - 2);
    return az[i] + (az[i + 1] - az[i]) * (s - k0 - i);
  };
  const a0 = at(start), a1 = at(start + span);
  return (u: number) => {
    if (Math.abs(a1 - a0) < 1e-9 || u <= 0 || u >= 1) return alongOrbit(cams, start + span * Math.min(1, Math.max(0, u)));
    const target = a0 + (a1 - a0) * u;
    let i = 0;
    while (i < az.length - 2 && dir * (az[i + 1] - target) < 0) i++;
    const seg = az[i + 1] - az[i];
    const f = Math.abs(seg) < 1e-12 ? 0 : Math.min(1, Math.max(0, (target - az[i]) / seg));
    return alongOrbit(cams, k0 + i + f);
  };
}

/** Leva a câmera para `radius` do centro da órbita, ao longo da reta que a
 *  liga ao centro (mesma direção de visão). Sem `radius`, fica onde está. */
function atRadius<T extends { center: Vec3; anchor: string | null }>(p: T, orbitC: Vec3, radius: number | undefined): T {
  if (!radius) return p;
  const r = sub(p.center, orbitC);
  const d = norm(r);
  if (d < 1e-9 || Math.abs(radius / d - 1) < 1e-9) return p;
  return { ...p, center: add(orbitC, scale(r, radius / d)), anchor: null };
}

/** Distância de uma pose ao centro da órbita das fotos. */
export function orbitDistance(cams: CameraPose[], pose: { qvec: Quat; tvec: Vec3 }): number {
  return norm(sub(toWorld(pose).center, orbitFrame(cams).center));
}

/** Pico da velocidade angular da câmera, em graus por segundo (média em
 *  janelas de `window` frames, para não pegar ruído de um frame só). */
export function peakSpeed(poses: { qvec: Quat }[], fps: number, window = 3): number {
  let peak = 0;
  for (let i = window; i < poses.length; i++) {
    const a = poses[i - window].qvec, b = poses[i].qvec;
    const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
    const deg = (2 * Math.acos(Math.min(1, d)) * 180) / Math.PI;
    peak = Math.max(peak, (deg * fps) / window);
  }
  return peak;
}

/** Câmera real mais próxima, em ângulo em torno do centro da órbita. */
export function nearestCamera(cams: CameraPose[], pose: { qvec: Quat; tvec: Vec3 }, center: Vec3): number {
  const dir = unit(sub(toWorld(pose).center, center));
  let best = 0, bestDot = -Infinity;
  cams.forEach((c, i) => {
    const d = dot(unit(sub(toWorld(c).center, center)), dir);
    if (d > bestDot) [best, bestDot] = [i, d];
  });
  return best;
}

export interface PresetOptions extends ClipOptions {
  /** Índice (em `cams`) da foto no meio do clipe. */
  center: number;
  /** Elevação máxima em graus em relação às fotos (negativo = abaixo). */
  elevDeg: number;
  /** Fotos percorridas; padrão (frames-1)/2, metade de uma foto por frame.
   *  Com o número de fotos, a câmera dá a volta inteira e termina na foto
   *  em que começou. */
  span?: number;
  /** "arc": desvia `elevDeg` no meio e volta; "wave": numa metade desvia
   *  `elevDeg`, na outra o oposto (sobe e desce). */
  shape?: "arc" | "wave";
  /** Distância ao centro da órbita do clipe inteiro; padrão, a das fotos. */
  radius?: number;
}

/** Atalho: percorre a órbita em volta de uma foto subindo ou descendo
 *  `elevDeg` no meio (rotação rígida em torno do centro). */
export function presetPath(cams: CameraPose[], lens: Lens, o: PresetOptions): ClipPlan {
  const { center: orbitC, up } = orbitFrame(cams);
  const n = o.frames;
  const span = o.span ?? (n - 1) / 2;
  const start = o.center - span / 2;
  const walk = orbitWalk(cams, orbitC, up, start, span);
  const poses: PathPose[] = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    const base = atRadius(walk(u), orbitC, o.radius);
    const ramp = profile(u, o.hold, o.ramp);
    // Na onda, o seno corre só no trecho fora das âncoras; a rampa suaviza a saída.
    const inner = Math.min(1, Math.max(0, (u - o.hold) / (1 - 2 * o.hold)));
    const w = o.shape === "wave" ? ramp * Math.sin(2 * Math.PI * inner) : ramp;
    let { rot, center } = base;
    const r = sub(center, orbitC);
    const radial = sub(r, scale(up, dot(r, up)));
    if (w !== 0 && norm(radial) > 1e-9) {
      // Eixo horizontal: ângulo positivo leva a câmera para cima.
      const R = axisAngle(cross(unit(radial), up), ((o.elevDeg * Math.PI) / 180) * w);
      center = add(orbitC, mulMV(R, r));
      rot = mulMM(R, rot);
    }
    poses.push({ ...fromWorld(rot, center), anchor: w === 0 ? base.anchor : null });
  }
  return { ...clipLens(lens, o.longSide, o.aspect), poses };
}

export interface ViewOptions extends ClipOptions {
  /** Fotos percorridas de ponta a ponta. */
  span: number;
  /** Varredura em azimute em volta da vista mirada, em graus (paralaxe). */
  sweepDeg: number;
}

/** Sai da direção de uma foto real, mistura até a vista mirada (com uma
 *  pequena varredura em volta dela) e termina na direção de outra foto. O
 *  clipe inteiro fica na distância da vista: sem aproximar nem afastar. */
export function viewPath(cams: CameraPose[], lens: Lens, target: { qvec: Quat; tvec: Vec3 }, o: ViewOptions): ClipPlan {
  const { center: orbitC, up } = orbitFrame(cams);
  const mid = nearestCamera(cams, target, orbitC);
  const start = mid - o.span / 2;
  const walk = orbitWalk(cams, orbitC, up, start, o.span);
  const t = toWorld(target);
  const radius = norm(sub(t.center, orbitC));
  const n = o.frames;
  const poses: PathPose[] = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    const base = atRadius(walk(u), orbitC, radius);
    const w = profile(u, o.hold, o.ramp);
    // Alvo girado em torno do "up" da órbita, de −sweep a +sweep.
    const R = axisAngle(up, (((u - 0.5) * 2 * o.sweepDeg) * Math.PI) / 180);
    const tc = add(orbitC, mulMV(R, sub(t.center, orbitC)));
    const trot = mulMM(R, t.rot);
    // Posição: mistura em coordenadas de órbita (azimute, elevação, raio) para
    // não cortar caminho por dentro do sujeito.
    const center = blendOrbit(base.center, tc, w, orbitC, up);
    const rot = transpose(quatToMat(slerp(matToQuat(transpose(base.rot)), matToQuat(transpose(trot)), w)));
    poses.push({ ...fromWorld(rot, center), anchor: w === 0 ? base.anchor : null });
  }
  return { ...clipLens(lens, o.longSide, o.aspect), poses };
}

/** Sentido da órbita filmada em volta de `up` (+1 ou −1), pela soma dos
 *  passos de azimute entre fotos vizinhas. */
function orbitSense(cams: CameraPose[], orbitC: Vec3, up: Vec3): number {
  const ref: Vec3 = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = unit(cross(up, ref));
  const e2 = cross(up, e1);
  const az = cams.map((c) => {
    const r = sub(toWorld(c).center, orbitC);
    return Math.atan2(dot(r, e2), dot(r, e1));
  });
  let sum = 0;
  for (let i = 1; i < az.length; i++) {
    const d = az[i] - az[i - 1];
    sum += d - 2 * Math.PI * Math.round(d / (2 * Math.PI));
  }
  return sum < 0 ? -1 : 1;
}

/** Volta 360° a partir de uma vista: gira a pose em torno do "up" da órbita,
 *  na mesma altura (ângulo), no sentido em que o vídeo foi filmado, e termina
 *  nela. Velocidade angular constante. `radius` leva a vista para outra
 *  distância do centro (mesma direção de visão); padrão, a da própria vista. */
export function turnPath(cams: CameraPose[], lens: Lens, target: { qvec: Quat; tvec: Vec3 }, o: ClipOptions & { radius?: number }): ClipPlan {
  const { center: orbitC, up } = orbitFrame(cams);
  const sense = orbitSense(cams, orbitC, up);
  const t = atRadius({ ...toWorld(target), anchor: null }, orbitC, o.radius);
  const n = o.frames;
  const poses: PathPose[] = [];
  for (let i = 0; i < n; i++) {
    const R = axisAngle(up, sense * 2 * Math.PI * (i / (n - 1)));
    poses.push({ ...fromWorld(mulMM(R, t.rot), add(orbitC, mulMV(R, sub(t.center, orbitC)))), anchor: null });
  }
  return { ...clipLens(lens, o.longSide, o.aspect), poses };
}

/** Interpola dois pontos em coordenadas esféricas em torno de `c` (eixo `up`). */
function blendOrbit(a: Vec3, b: Vec3, w: number, c: Vec3, up: Vec3): Vec3 {
  if (w <= 0) return a;
  if (w >= 1) return b;
  // Base ortonormal: e1, e2 no plano da órbita, up.
  const ref = Math.abs(up[0]) < 0.9 ? ([1, 0, 0] as Vec3) : ([0, 1, 0] as Vec3);
  const e1 = unit(cross(up, ref));
  const e2 = cross(up, e1);
  const sph = (p: Vec3) => {
    const r = sub(p, c);
    const rad = norm(r) || 1e-9;
    return { az: Math.atan2(dot(r, e2), dot(r, e1)), el: Math.asin(Math.max(-1, Math.min(1, dot(r, up) / rad))), rad };
  };
  const pa = sph(a), pb = sph(b);
  let daz = pb.az - pa.az;
  daz -= 2 * Math.PI * Math.round(daz / (2 * Math.PI));
  const az = pa.az + daz * w, el = pa.el + (pb.el - pa.el) * w, rad = pa.rad + (pb.rad - pa.rad) * w;
  const ce = Math.cos(el);
  return add(c, add(scale(e1, rad * ce * Math.cos(az)), add(scale(e2, rad * ce * Math.sin(az)), scale(up, rad * Math.sin(el)))));
}

/** Câmera → mundo de uma pose de trajetória (para o viewer). */
export const poseToWorld = toWorld;
