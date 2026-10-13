// Orbit camera for the link viewer: drag rotates, wheel/pinch zooms, right
// button/Shift or two fingers pan, and WASD/arrows move through the scene.
// Port of the WebGaussian player (orbit-camera.ts), with the pitch range locked
// to what the video covered, plus auto-spin.

import { Entity, Vec3 } from "playcanvas";

const ROTATE_SPEED = 0.25; // degrees per pixel
const PAN_SPEED = 0.0015; // fraction of the distance per pixel
const KEY_MOVE_SPEED = 1.2; // fraction of the distance per second
const SPIN_SPEED = 12; // degrees per second
const PITCH_LIMIT = 89;

const CAMERA_KEYS = new Set(["w", "a", "s", "d", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]);
const normalizeKey = (key: string) => (key.length === 1 ? key.toLowerCase() : key);

export class OrbitCamera {
  yaw = 0;
  pitch = -10;
  distance = 3;
  readonly target = new Vec3();
  spin = false;
  /** Pitch range in degrees; outside it the video captured nothing. */
  private minPitch = -PITCH_LIMIT;
  private maxPitch = PITCH_LIMIT;
  private minDistance = 0.01;
  private maxDistance = 1000;

  private readonly pointers = new Map<number, { x: number; y: number }>();
  private lastPinch = 0;
  private lastMid: { x: number; y: number } | null = null;
  private readonly keys = new Set<string>();
  private shift = false;
  private readonly abort = new AbortController();
  /** Called when the user interacts (to stop the auto-spin). */
  onInteract: () => void = () => {};

  constructor(
    private readonly camera: Entity,
    private readonly dom: HTMLElement,
  ) {
    const opts: AddEventListenerOptions = { signal: this.abort.signal };
    dom.addEventListener("pointerdown", (e) => this.onPointerDown(e), opts);
    dom.addEventListener("pointermove", (e) => this.onPointerMove(e), opts);
    dom.addEventListener("pointerup", (e) => this.onPointerUp(e), opts);
    dom.addEventListener("pointercancel", (e) => this.onPointerUp(e), opts);
    dom.addEventListener("wheel", (e) => this.onWheel(e), { ...opts, passive: false });
    dom.addEventListener("contextmenu", (e) => e.preventDefault(), opts);
    dom.addEventListener("keydown", (e) => this.onKey(e, true), opts);
    dom.addEventListener("keyup", (e) => this.onKey(e, false), opts);
    // A key stuck "down" after focus is lost would keep the camera moving.
    dom.addEventListener("blur", () => this.keys.clear(), opts);
    this.apply();
  }

  /** Allowed pitch, in degrees (negative looks down). */
  setPitchRange(min: number, max: number) {
    this.minPitch = Math.max(-PITCH_LIMIT, min);
    this.maxPitch = Math.min(PITCH_LIMIT, max);
    this.apply();
  }

  setDistanceRange(min: number, max: number) {
    this.minDistance = min;
    this.maxDistance = max;
    this.apply();
  }

  /** Looks at `target` from `position`. */
  lookFrom(position: Vec3, target: Vec3) {
    const d = new Vec3().sub2(position, target);
    this.target.copy(target);
    this.distance = d.length() || 1;
    // The camera looks along -d; for PlayCanvas orientation (pitch, yaw, 0),
    // the view direction is (-cos p · sin y, sin p, -cos p · cos y).
    this.pitch = (Math.asin(Math.max(-1, Math.min(1, -d.y / this.distance))) * 180) / Math.PI;
    this.yaw = (Math.atan2(d.x, d.z) * 180) / Math.PI;
    this.apply();
  }

  /** Advances auto-spin and keyboard movement; call every frame. */
  update(dt: number) {
    if (dt <= 0) return;
    if (this.spin) {
      this.yaw += SPIN_SPEED * dt;
      this.apply();
    }
    if (!this.keys.size) return;
    const has = (...k: string[]) => k.some((key) => this.keys.has(key));
    const forward = (has("w", "ArrowUp") ? 1 : 0) - (has("s", "ArrowDown") ? 1 : 0);
    const strafe = (has("d", "ArrowRight") ? 1 : 0) - (has("a", "ArrowLeft") ? 1 : 0);
    if (!forward && !strafe) return;
    // Moves the target along (distance unchanged), with a step proportional
    // to the scene size.
    const step = this.distance * KEY_MOVE_SPEED * dt;
    if (this.shift) this.target.y += forward * step;
    else this.target.add(this.camera.forward.clone().mulScalar(forward * step));
    this.target.add(this.camera.right.clone().mulScalar(strafe * step));
    this.apply();
  }

  dispose() {
    this.abort.abort();
  }

  private onKey(e: KeyboardEvent, down: boolean) {
    const key = normalizeKey(e.key);
    if (!CAMERA_KEYS.has(key)) return;
    e.preventDefault();
    if (down) this.keys.add(key);
    else this.keys.delete(key);
    this.shift = e.shiftKey;
    this.onInteract();
  }

  private onPointerDown(e: PointerEvent) {
    // Grab keyboard focus so WASD works right after a click.
    this.dom.focus({ preventScroll: true });
    this.dom.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    this.lastPinch = 0;
    this.lastMid = null;
    this.onInteract();
  }

  private onPointerUp(e: PointerEvent) {
    this.pointers.delete(e.pointerId);
    this.lastPinch = 0;
    this.lastMid = null;
  }

  private onPointerMove(e: PointerEvent) {
    const prev = this.pointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (this.pointers.size === 2) {
      // Two fingers: pinch zooms, dragging both pans.
      const [a, b] = [...this.pointers.values()];
      const pinch = Math.hypot(a.x - b.x, a.y - b.y);
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      if (this.lastPinch > 0) this.zoom(this.lastPinch / pinch);
      if (this.lastMid) this.pan(mid.x - this.lastMid.x, mid.y - this.lastMid.y);
      this.lastPinch = pinch;
      this.lastMid = mid;
    } else if (e.buttons & 6 || e.shiftKey) {
      this.pan(dx, dy);
    } else {
      this.yaw -= dx * ROTATE_SPEED;
      this.pitch -= dy * ROTATE_SPEED;
    }
    this.apply();
  }

  private onWheel(e: WheelEvent) {
    e.preventDefault();
    this.zoom(Math.exp(e.deltaY * 0.0015));
    this.apply();
    this.onInteract();
  }

  private zoom(factor: number) {
    this.distance *= factor;
  }

  private pan(dx: number, dy: number) {
    const scale = this.distance * PAN_SPEED;
    const right = this.camera.right;
    const up = this.camera.up;
    this.target.x += (-dx * right.x + dy * up.x) * scale;
    this.target.y += (-dx * right.y + dy * up.y) * scale;
    this.target.z += (-dx * right.z + dy * up.z) * scale;
  }

  private apply() {
    this.pitch = Math.min(this.maxPitch, Math.max(this.minPitch, this.pitch));
    this.distance = Math.min(this.maxDistance, Math.max(this.minDistance, this.distance));
    this.camera.setEulerAngles(this.pitch, this.yaw, 0);
    const f = this.camera.forward;
    this.camera.setPosition(
      this.target.x - f.x * this.distance,
      this.target.y - f.y * this.distance,
      this.target.z - f.z * this.distance,
    );
  }
}
