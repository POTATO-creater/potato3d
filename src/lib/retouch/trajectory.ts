// Retouch trajectories, in COLMAP convention (pose = world → camera, camera
// looking down +Z with +Y pointing down). They leave from a real photo, pass
// through angles the orbit never saw and return to a real photo: the ends
// (anchors) have exactly a photo's pose, which gives the video model a real
// reference and lets us measure the restoration. Port of build_trajectory from
// scripts/tests/restore-test/restore_test.py.

export type Vec3 = [number, number, number];
/** (w, x, y, z), COLMAP order. */
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
  /** Photo name when the pose is exactly that photo's. */
  anchor: string | null;
}

export interface ClipPlan {
  size: [number, number];
  intrinsics: { fx: number; fy: number; cx: number; cy: number };
  poses: PathPose[];
}

export interface ClipOptions {
  /** Clip frame count; LTX wants 8n + 1. */
  frames: number;
  /** Video long side, rounded (along with the short side) to multiples of 32. */
  longSide: number;
  /** Fraction of the clip held still at each end (anchors). */
  hold: number;
  /** Fraction of the clip spent in each transition. */
  ramp: number;
  /** Width/height aspect ratio; defaults to the lens's. */
  aspect?: number;
  /** Point to move around instead of the photos' center (the viewer's
   *  anchor), in COLMAP coordinates. */
  pivot?: Vec3;
}

/** No held ends: each transition uses 40% of the clip, which makes the camera
 *  much slower than with held ends (LTX blurs anything that moves fast). */
export const DEFAULT_CLIP: ClipOptions = { frames: 97, longSide: 960, hold: 0, ramp: 0.4 };

/** Frame count for a clip of `seconds` at `fps`, at the nearest 8n + 1 (LTX
 *  works in blocks of 8 frames plus the first). Minimum 17. */
export function clipFrames(seconds: number, fps: number): number {
  const n = Math.round((seconds * fps - 1) / 8);
  return Math.max(2, n) * 8 + 1;
}

// ---------------------------------------------------------------- algebra --

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const unit = (a: Vec3): Vec3 => scale(a, 1 / (norm(a) || 1));
/** Angle between two rotations, in radians. */
const quatAngle = (a: Quat, b: Quat) => 2 * Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])));
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

/** Rotation of `ang` radians about `axis` (Rodrigues). */
function axisAngle(axis: Vec3, ang: number): Mat3 {
  const [x, y, z] = unit(axis);
  const c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
  return [
    [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
  ];
}

/** Camera → world pose: rotation and center. */
const toWorld = (p: { qvec: Quat; tvec: Vec3 }) => {
  const r = quatToMat(p.qvec);
  const rt = transpose(r);
  return { rot: rt, center: scale(mulMV(rt, p.tvec), -1) };
};
const fromWorld = (rot: Mat3, center: Vec3): { qvec: Quat; tvec: Vec3 } => {
  const r = transpose(rot);
  return { qvec: matToQuat(r), tvec: scale(mulMV(r, center), -1) };
};

/** Orbit center and "up". Center: the point closest to all optical axes
 *  (where the cameras look); up: mean of the cameras' -Y. */
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

// ----------------------------------------------------------- trajectories --

/** 0 at the ends (anchors), 1 in the middle, with cosine ramps. */
export function profile(u: number, hold: number, ramp: number): number {
  const d = Math.min(u, 1 - u);
  if (d <= hold) return 0;
  if (d >= hold + ramp) return 1;
  return 0.5 - 0.5 * Math.cos((Math.PI * (d - hold)) / ramp);
}

/** Clip size (multiples of 32, long side ~longSide) and intrinsics with the
 *  principal point at the center, like the viewer's render. Without `aspect`
 *  (width/height) the lens's aspect is used; with it, the lens's vertical
 *  field of view is kept and the aspect only changes the frame width. */
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
  // Scale that covers both sides; the excess is cropped equally at the edges.
  const k = Math.max(w / lens.width, h / lens.height);
  return { size: [w, h], intrinsics: { fx: lens.fx * k, fy: lens.fy * k, cx: w / 2, cy: h / 2 } };
}

/** Pose interpolated along the photos (fractional index, closed orbit). */
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

/** Walks `span` photos from index `start` at constant angular speed around
 *  the center. Photos are almost never evenly spaced (and in a closed orbit
 *  the last one usually repeats the first): stepping "one photo per frame"
 *  makes the camera speed up, slow down and even stop. */
function orbitWalk(cams: CameraPose[], orbitC: Vec3, up: Vec3, start: number, span: number) {
  const m = cams.length;
  const ref: Vec3 = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = unit(cross(up, ref));
  const e2 = cross(up, e1);
  const k0 = Math.floor(start);
  const k1 = Math.ceil(start + span);
  // Unwrapped azimuth at each photo in the segment.
  const az: number[] = [];
  for (let k = k0; k <= k1; k++) {
    const r = sub(toWorld(cams[((k % m) + m) % m]).center, orbitC);
    let a = Math.atan2(dot(r, e2), dot(r, e1));
    if (az.length) a -= 2 * Math.PI * Math.round((a - az[az.length - 1]) / (2 * Math.PI));
    az.push(a);
  }
  // Monotonic in the direction of travel (pose noise must not make it go back).
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

/** Moves the camera to `radius` from the orbit center, along the line joining
 *  it to the center (same view direction). Without `radius`, it stays put. */
function atRadius<T extends { center: Vec3; anchor: string | null }>(p: T, orbitC: Vec3, radius: number | undefined): T {
  if (!radius) return p;
  const r = sub(p.center, orbitC);
  const d = norm(r);
  if (d < 1e-9 || Math.abs(radius / d - 1) < 1e-9) return p;
  return { ...p, center: add(orbitC, scale(r, radius / d)), anchor: null };
}

/** Distance from a pose to `center` (default: the photos' orbit center). */
export function orbitDistance(cams: CameraPose[], pose: { qvec: Quat; tvec: Vec3 }, center?: Vec3): number {
  return norm(sub(toWorld(pose).center, center ?? orbitFrame(cams).center));
}

/** Centers for a move: `walk`, the photos' orbit center, which times walks
 *  along the photos (by their azimuth); `center`, what the move turns around:
 *  the pivot when there is one, otherwise the same. */
function moveFrame(cams: CameraPose[], pivot?: Vec3) {
  const { center, up } = orbitFrame(cams);
  return { walk: center, center: pivot ?? center, up };
}

/** Turns a camera (camera → world rotation) by fraction `w` of the way
 *  toward aiming at `target` (the shortest rotation of its optical axis). */
function aimToward(rot: Mat3, from: Vec3, target: Vec3, w: number): Mat3 {
  if (w <= 0) return rot;
  const fwd: Vec3 = [rot[0][2], rot[1][2], rot[2][2]];
  const dir = unit(sub(target, from));
  const axis = cross(fwd, dir);
  const s = norm(axis);
  if (s < 1e-9) return rot;
  return mulMM(axisAngle(axis, Math.atan2(s, dot(fwd, dir)) * Math.min(1, w)), rot);
}

/** Peak camera angular speed in degrees per second (averaged over windows of
 *  `window` frames, to ignore single-frame noise). With `center`, travel
 *  counts too (see `viewChange`): a zoom rotates nothing but changes the
 *  image as much as a turn. */
export function peakSpeed(poses: { qvec: Quat; tvec: Vec3 }[], fps: number, window = 3, center?: Vec3): number {
  const change = center ? viewChange(poses, center) : null;
  let peak = 0;
  for (let i = window; i < poses.length; i++) {
    const rad = change ? change(i - window, i) : quatAngle(poses[i - window].qvec, poses[i].qvec);
    peak = Math.max(peak, (((rad * 180) / Math.PI) * fps) / window);
  }
  return peak;
}

/** How much the image changes between two poses, as an angle in radians:
 *  the larger of the camera's rotation and its travel over its distance to
 *  `center`. Orbiting the center, both match the angle turned; zooming, the
 *  travel over the distance is the change in image scale. The distance has a
 *  floor (5% of the median) so passing through the center doesn't explode. */
function viewChange(poses: { qvec: Quat; tvec: Vec3 }[], center: Vec3) {
  const centers = poses.map((p) => toWorld(p).center);
  const dist = centers.map((c) => norm(sub(c, center)));
  const floor = 0.05 * ([...dist].sort((a, b) => a - b)[dist.length >> 1] ?? 0) || 1e-9;
  return (i: number, j: number) =>
    Math.max(quatAngle(poses[i].qvec, poses[j].qvec), norm(sub(centers[j], centers[i])) / Math.max(floor, (dist[i] + dist[j]) / 2));
}

/** Nearest real camera, by angle around the orbit center. */
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
  /** Index (into `cams`) of the photo at the middle of the clip. */
  center: number;
  /** Maximum elevation in degrees relative to the photos (negative = below). */
  elevDeg: number;
  /** Photos traversed; default (frames-1)/2, half a photo per frame. With the
   *  photo count, the camera goes all the way around and ends at the photo
   *  it started from. */
  span?: number;
  /** "arc": deviates `elevDeg` in the middle and comes back; "wave": deviates
   *  `elevDeg` in one half and the opposite in the other (up and down). */
  shape?: "arc" | "wave";
  /** Distance to the orbit center for the whole clip; defaults to the photos'. */
  radius?: number;
}

/** Preset: walks the orbit around a photo, rising or dropping `elevDeg` in
 *  the middle (rigid rotation about the center). */
export function presetPath(cams: CameraPose[], lens: Lens, o: PresetOptions): ClipPlan {
  const { walk: walkC, center: orbitC, up } = moveFrame(cams, o.pivot);
  const n = o.frames;
  const span = o.span ?? (n - 1) / 2;
  const start = o.center - span / 2;
  const walk = orbitWalk(cams, walkC, up, start, span);
  const poses: PathPose[] = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    const base = atRadius(walk(u), orbitC, o.radius);
    const ramp = profile(u, o.hold, o.ramp);
    // For the wave, the sine runs only outside the anchors; the ramp smooths the exit.
    const inner = Math.min(1, Math.max(0, (u - o.hold) / (1 - 2 * o.hold)));
    const w = o.shape === "wave" ? ramp * Math.sin(2 * Math.PI * inner) : ramp;
    let { rot, center } = base;
    const r = sub(center, orbitC);
    const radial = sub(r, scale(up, dot(r, up)));
    if (w !== 0 && norm(radial) > 1e-9) {
      // Horizontal axis: a positive angle moves the camera up.
      const R = axisAngle(cross(unit(radial), up), ((o.elevDeg * Math.PI) / 180) * w);
      center = add(orbitC, mulMV(R, r));
      rot = mulMM(R, rot);
    }
    // With a pivot, the camera turns to aim at it away from the photos (which
    // aim at their own center). The wave has no ramp: give the aim one.
    const aim = o.pivot ? profile(u, o.hold, o.ramp || 0.15) : 0;
    rot = aimToward(rot, center, orbitC, aim);
    poses.push({ ...fromWorld(rot, center), anchor: w === 0 && aim === 0 ? base.anchor : null });
  }
  return { ...clipLens(lens, o.longSide, o.aspect), poses };
}

export interface ViewOptions extends ClipOptions {
  /** Photos traversed end to end. */
  span: number;
  /** Azimuth sweep around the target view, in degrees (parallax). */
  sweepDeg: number;
}

/** Leaves from a real photo's direction, blends into the target view (with a
 *  small sweep around it) and ends in another photo's direction. The whole
 *  clip stays at the view's distance: no moving in or out. */
export function viewPath(cams: CameraPose[], lens: Lens, target: { qvec: Quat; tvec: Vec3 }, o: ViewOptions): ClipPlan {
  const { walk: walkC, center: orbitC, up } = moveFrame(cams, o.pivot);
  const mid = nearestCamera(cams, target, orbitC);
  const start = mid - o.span / 2;
  const walk = orbitWalk(cams, walkC, up, start, o.span);
  const t = toWorld(target);
  const radius = norm(sub(t.center, orbitC));
  const n = o.frames;
  const poses: PathPose[] = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    const base = atRadius(walk(u), orbitC, radius);
    const w = profile(u, o.hold, o.ramp);
    // Target rotated about the orbit's "up", from −sweep to +sweep.
    const R = axisAngle(up, (((u - 0.5) * 2 * o.sweepDeg) * Math.PI) / 180);
    const tc = add(orbitC, mulMV(R, sub(t.center, orbitC)));
    const trot = mulMM(R, t.rot);
    // Position: blend in orbit coordinates (azimuth, elevation, radius) so the
    // path doesn't cut through the subject.
    const center = blendOrbit(base.center, tc, w, orbitC, up);
    const rot = transpose(quatToMat(slerp(matToQuat(transpose(base.rot)), matToQuat(transpose(trot)), w)));
    poses.push({ ...fromWorld(rot, center), anchor: w === 0 ? base.anchor : null });
  }
  return { ...clipLens(lens, o.longSide, o.aspect), poses };
}

/** Direction of the filmed orbit around `up` (+1 or −1), from the sum of
 *  azimuth steps between neighbouring photos. */
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

/** 360° turn from a view: rotates the pose about the orbit's "up" at the same
 *  height (angle), in the direction the video was filmed, ending where it
 *  started. Constant angular speed. `radius` moves the view to another
 *  distance from the center (same view direction); defaults to the view's own. */
export function turnPath(cams: CameraPose[], lens: Lens, target: { qvec: Quat; tvec: Vec3 }, o: ClipOptions & { radius?: number }): ClipPlan {
  const { walk: walkC, center: orbitC, up } = moveFrame(cams, o.pivot);
  const sense = orbitSense(cams, walkC, up);
  const t = atRadius({ ...toWorld(target), anchor: null }, orbitC, o.radius);
  const n = o.frames;
  const poses: PathPose[] = [];
  for (let i = 0; i < n; i++) {
    const R = axisAngle(up, sense * 2 * Math.PI * (i / (n - 1)));
    poses.push({ ...fromWorld(mulMM(R, t.rot), add(orbitC, mulMV(R, sub(t.center, orbitC)))), anchor: null });
  }
  return { ...clipLens(lens, o.longSide, o.aspect), poses };
}


/** Rigid rotation of a pose (camera → world) about `orbitC`. */
const rotateAbout = (p: { rot: Mat3; center: Vec3 }, R: Mat3, orbitC: Vec3) => ({
  rot: mulMM(R, p.rot),
  center: add(orbitC, mulMV(R, sub(p.center, orbitC))),
});

/** Horizontal axis for raising or lowering a camera (positive angle = up):
 *  the camera's own "right", flattened onto the orbit plane. With the axis
 *  taken from the position (the radial), a near-vertical view would come down
 *  on the wrong side and the image would flip upside down, because the
 *  viewer's aim point is not exactly the orbit center. */
function pitchAxis(p: { rot: Mat3; center: Vec3 }, orbitC: Vec3, up: Vec3): Vec3 | null {
  const right: Vec3 = [p.rot[0][0], p.rot[1][0], p.rot[2][0]];
  const axis = sub(right, scale(up, dot(right, up)));
  if (norm(axis) >= 1e-6) return scale(unit(axis), -1);
  // Camera on its side (rolled 90°): fall back to the radial.
  const r = sub(p.center, orbitC);
  const radial = sub(r, scale(up, dot(r, up)));
  return norm(radial) < 1e-9 ? null : cross(unit(radial), up);
}

/** Point the camera aims at: where the optical axis passes closest to the
 *  subject's vertical axis (the line through `orbitC` along `up`). When looking
 *  nearly vertically (parallel axes), the optical-axis point closest to the center. */
function focusPoint(c: Vec3, fwd: Vec3, orbitC: Vec3, up: Vec3): Vec3 {
  const w = sub(c, orbitC);
  const b = dot(fwd, up);
  const den = 1 - b * b;
  const minT = 0.2 * norm(w);
  let t = den < 1e-4 ? -dot(w, fwd) : (b * dot(w, up) - dot(w, fwd)) / den;
  if (!(t > minT)) t = Math.max(minT, -dot(w, fwd));
  return add(c, scale(fwd, t));
}

const ease = (x: number) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, x)));

/** Rotation (camera → world) of a level camera, with no roll, facing the same
 *  way as `rot` (its forward on the horizontal plane). When looking nearly
 *  vertically, the heading comes from the image's "up": it points forward when
 *  the camera looks down and backward when it looks up. */
function levelRotation(rot: Mat3, up: Vec3): Mat3 {
  const flat = (v: Vec3) => sub(v, scale(up, dot(v, up)));
  const fwd: Vec3 = [rot[0][2], rot[1][2], rot[2][2]];
  const imgUp: Vec3 = [-rot[0][1], -rot[1][1], -rot[2][1]];
  const pitch = dot(fwd, up);
  const face = unit(add(flat(fwd), scale(flat(imgUp), -Math.sign(pitch))));
  const down = scale(up, -1);
  const right = cross(down, face);
  return [
    [right[0], down[0], face[0]],
    [right[1], down[1], face[1]],
    [right[2], down[2], face[2]],
  ];
}

/** 360° turn at a view's tilt, in three phases:
 *  1. straight lift: starts at the photos' height (the first frame, the video
 *     model's reference, sits at a well-covered height) with the camera level,
 *     at the view's direction and distance, and rises (or drops) to the view
 *     along an arc around the point it aims at, like a crane; the camera
 *     gradually tilts from horizontal to the view's tilt;
 *  2. a 360° turn at that tilt, about the vertical axis through the aim point;
 *  3. straight descent back to the first frame.
 *  Each phase's duration is proportional to the angle covered, and the camera
 *  slows down at phase changes (no corners). */
export function tiltTurnPath(cams: CameraPose[], lens: Lens, target: { qvec: Quat; tvec: Vec3 }, o: ClipOptions & { radius?: number }): ClipPlan {
  const { walk: walkC, center: orbitC, up } = moveFrame(cams, o.pivot);
  const sense = orbitSense(cams, walkC, up);
  const t = atRadius({ ...toWorld(target), anchor: null }, orbitC, o.radius);
  const fwd: Vec3 = [t.rot[0][2], t.rot[1][2], t.rot[2][2]];
  const focus = focusPoint(t.center, fwd, orbitC, up);
  const axis = pitchAxis(t, orbitC, up);
  const r = sub(t.center, focus);
  const rho = norm(r);
  // Camera elevation around the aim point, in the lift plane.
  const elView = axis ? Math.atan2(dot(r, up), dot(r, cross(up, axis))) : Math.PI / 2;
  // Start at the photos' mean height (clamped to ±30° around the aim point).
  const photoH = cams.reduce((a, c) => a + dot(sub(toWorld(c).center, orbitC), up), 0) / cams.length;
  const focusH = dot(sub(focus, orbitC), up);
  const elStart = Math.asin(Math.max(-0.5, Math.min(0.5, (photoH - focusH) / (rho || 1))));
  const delta = elView - elStart;
  const crane = (ang: number) => (axis && Math.abs(ang) > 1e-12 ? rotateAbout(t, axisAngle(axis, ang), focus) : t);
  // During the lift, position follows the arc and rotation goes from level to the view's.
  const levelQ = matToQuat(transpose(levelRotation(t.rot, up)));
  const viewQ = matToQuat(transpose(t.rot));
  const lifting = (e: number) => ({ center: crane(-delta * (1 - e)).center, rot: transpose(quatToMat(slerp(levelQ, viewQ, e))) });
  // Clip fractions: lift and descent proportional to the angle (10% to 30%).
  const lift = Math.min(0.3, Math.max(0.1, Math.abs(delta) / (2 * Math.abs(delta) + 2 * Math.PI)));
  const n = o.frames;
  const poses: PathPose[] = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    let p;
    if (u < lift) p = lifting(ease(u / lift));
    else if (u <= 1 - lift) p = rotateAbout(t, axisAngle(up, sense * 2 * Math.PI * ease((u - lift) / (1 - 2 * lift))), focus);
    else p = lifting(1 - ease((u - (1 - lift)) / lift));
    poses.push({ ...fromWorld(p.rot, p.center), anchor: null });
  }
  return { ...clipLens(lens, o.longSide, o.aspect), poses };
}

/** Viewer camera sampled while recording a motion. */
export interface RecordedSample {
  /** Milliseconds (any origin; only differences matter). */
  t: number;
  qvec: Quat;
  tvec: Vec3;
}

/** Recorded time span with movement: the pauses before the first move and
 *  after the last one are cut, keeping the last still sample before the move
 *  (and the first one after) so the clip starts and ends at rest. Null when the
 *  camera never moved. */
export function recordedSpan(samples: RecordedSample[]): [number, number] | null {
  if (samples.length < 2) return null;
  const world = samples.map((s) => toWorld(s));
  // Thresholds relative to the size of the movement: OrbitControls damping
  // leaves a tail of tiny moves after the last input, which counts as stillness.
  let extent = 0;
  for (const w of world) extent = Math.max(extent, norm(sub(w.center, world[0].center)), norm(sub(w.center, world[world.length - 1].center)));
  const posEps = Math.max(extent * 2e-3, 1e-9);
  const angEps = (0.1 * Math.PI) / 180;
  const moved = (i: number, j: number, k = 1) =>
    norm(sub(world[i].center, world[j].center)) > posEps * k || quatAngle(samples[i].qvec, samples[j].qvec) > angEps * k;
  if (!samples.some((_, i) => moved(i, 0))) return null;
  // Before the first input the camera is exactly still: strict thresholds, so
  // a slow start is not cut.
  let first = 1;
  while (!moved(first, 0, 1e-3)) first++;
  const n = samples.length - 1;
  let last = n - 1;
  while (last > first && !moved(last, n)) last--;
  return [first - 1, last + 1];
}

/** Recorded duration with movement, in seconds (0 when nothing moved). */
export function recordedSeconds(samples: RecordedSample[]): number {
  const span = recordedSpan(samples);
  return span ? (samples[span[1]].t - samples[span[0]].t) / 1000 : 0;
}

/** The camera barely moved between two samples (OrbitControls damping keeps
 *  nudging it by vanishing amounts long after the last input). */
export function poseStill(a: { qvec: Quat; tvec: Vec3 }, b: { qvec: Quat; tvec: Vec3 }): boolean {
  const scaleT = Math.max(norm(a.tvec), norm(b.tvec), 1e-9);
  return quatAngle(a.qvec, b.qvec) < (1e-3 * Math.PI) / 180 && norm(sub(toWorld(a).center, toWorld(b).center)) < 1e-5 * scaleT;
}

export interface RecordedOptions extends ClipOptions {
  /** 0 keeps the recorded pace; 1 moves at constant speed along the recorded
   *  path (pauses in the middle disappear); in between, a blend. */
  uniform?: number;
  /** 0 keeps the recorded shake; 1 smooths the path with a Gaussian of sigma
   *  10% of the recording (position and aim). The ends stay where recorded. */
  stabilize?: number;
  /** Orbit center: the path is measured with `viewChange` around it. */
  center: Vec3;
}

/** Point of a path: camera center and world → camera rotation. */
interface TrackPoint {
  center: Vec3;
  q: Quat;
}

/** Points of `pts` at curve parameters `targets` (ascending), where `param`
 *  (non-decreasing) is each point's parameter. Linear in position, slerp in
 *  rotation. */
function resample(pts: TrackPoint[], param: number[], targets: number[]): TrackPoint[] {
  let k = 0;
  return targets.map((p) => {
    while (k < pts.length - 2 && param[k + 1] < p) k++;
    const dp = param[k + 1] - param[k];
    const f = dp > 0 ? Math.min(1, Math.max(0, (p - param[k]) / dp)) : 1;
    const a = pts[k], b = pts[k + 1];
    return { center: add(scale(a.center, 1 - f), scale(b.center, f)), q: slerp(a.q, b.q, f) };
  });
}

/** Moving average of radius `r` with odd reflection at the ends (the ends
 *  extend their own trend), which keeps the first and last values exactly. */
function boxPass(v: Vec3[], r: number): Vec3[] {
  const m = v.length;
  const at = (i: number): Vec3 => (i < 0 ? sub(scale(v[0], 2), v[-i]) : i >= m ? sub(scale(v[m - 1], 2), v[2 * (m - 1) - i]) : v[i]);
  const sums: Vec3[] = [[0, 0, 0]];
  for (let i = -r; i < m + r; i++) sums.push(add(sums[sums.length - 1], at(i)));
  return v.map((_, j) => scale(sub(sums[j + 2 * r + 1], sums[j]), 1 / (2 * r + 1)));
}

/** Gaussian blur of `sigma` samples (three moving averages). */
function blur(v: Vec3[], sigma: number): Vec3[] {
  const r = Math.min(v.length - 1, Math.round(Math.sqrt(4 * sigma * sigma + 1) / 2));
  return r < 1 ? v : boxPass(boxPass(boxPass(v, r), r), r);
}

/** Smooths a path evenly spaced in time: position, and the aim through the
 *  camera's forward and down axes, which are then made orthonormal again. */
function stabilizeTrack(pts: TrackPoint[], sigma: number): TrackPoint[] {
  const rots = pts.map((p) => transpose(quatToMat(p.q)));
  const col = (m: Mat3, j: number): Vec3 => [m[0][j], m[1][j], m[2][j]];
  const centers = blur(pts.map((p) => p.center), sigma);
  const fwd = blur(rots.map((m) => col(m, 2)), sigma);
  const down = blur(rots.map((m) => col(m, 1)), sigma);
  return pts.map((_, i) => {
    const f = unit(fwd[i]);
    const r = unit(cross(down[i], f));
    const d = cross(f, r);
    const c2w: Mat3 = [
      [r[0], d[0], f[0]],
      [r[1], d[1], f[1]],
      [r[2], d[2], f[2]],
    ];
    return { center: centers[i], q: matToQuat(transpose(c2w)) };
  });
}

/** Trajectory recorded from the viewer camera: the moving part of the
 *  recording (see `recordedSpan`) on a dense, evenly timed grid, smoothed by
 *  `stabilize`, then resampled at `frames` instants. With `uniform` 0 the
 *  instants are evenly spaced in time, so the camera keeps the pace it was
 *  recorded at (stretched to the clip's duration); with 1 they are evenly
 *  spaced along the (smoothed) path, measured by how much the image changes
 *  (`viewChange`), for a constant speed, zooms included. */
export function recordedPath(samples: RecordedSample[], lens: Lens, o: RecordedOptions): ClipPlan | null {
  const span = recordedSpan(samples);
  if (!span) return null;
  const seg = samples.slice(span[0], span[1] + 1);
  const n = o.frames;
  // Dense grid: finer than both the recording and the clip.
  const m = Math.min(4096, Math.max(256, 2 * seg.length, 4 * n));
  const t0 = seg[0].t, dur = seg[seg.length - 1].t - t0 || 1;
  const even = (count: number) => Array.from({ length: count }, (_, i) => i / (count - 1));
  let dense = resample(
    seg.map((s) => ({ center: toWorld(s).center, q: s.qvec })),
    seg.map((s) => (s.t - t0) / dur),
    even(m),
  );
  const stab = Math.min(1, Math.max(0, o.stabilize ?? 0));
  if (stab > 0) dense = stabilizeTrack(dense, stab * 0.1 * (m - 1));
  // Curve parameter of each point, from 0 to 1: time, path length or a blend.
  const change = viewChange(dense.map((p) => ({ qvec: p.q, tvec: fromWorld(transpose(quatToMat(p.q)), p.center).tvec })), o.center);
  const len = [0];
  for (let k = 1; k < m; k++) len.push(len[k - 1] + change(k - 1, k));
  const total = len[m - 1];
  const w = total > 1e-9 ? Math.min(1, Math.max(0, o.uniform ?? 0)) : 0;
  const param = len.map((l, k) => (1 - w) * (k / (m - 1)) + (w * l) / (total || 1));
  const poses: PathPose[] = resample(dense, param, even(n)).map((p) => ({
    ...fromWorld(transpose(quatToMat(p.q)), p.center),
    anchor: null,
  }));
  return { ...clipLens(lens, o.longSide, o.aspect), poses };
}

/** Interpolates two points in spherical coordinates around `c` (axis `up`). */
function blendOrbit(a: Vec3, b: Vec3, w: number, c: Vec3, up: Vec3): Vec3 {
  if (w <= 0) return a;
  if (w >= 1) return b;
  // Orthonormal basis: e1, e2 in the orbit plane, up.
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

/** Camera → world of a trajectory pose (for the viewer). */
export const poseToWorld = toWorld;
