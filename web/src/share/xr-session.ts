// WebXR session for the link viewer (VR on headsets, AR on phones):
// availability, enter/exit, placing the scene and moving with the controllers.
// Port of the WebGaussian player (xr-session.ts).
//
// The scene itself is untouched; two nodes created with the app do the work:
// - `rig`: the camera's parent. The headset pose is composed with it, so
//   moving (thumbstick) and turning (snap turn) move the rig.
// - `stage`: the splat's parent. On enter it scales the scene into a ~1.4 m
//   diorama standing in front of the user, turned toward the side the video
//   covered; on exit it goes back to identity for the orbit camera.
//
// Unlike WebGaussian (a single isolated object), here the scene has a
// background and stray splats around the cameras. On a headset that hurts:
// large splats covering the whole view, twice (once per eye), are the biggest
// cost on a Quest, and up close they turn into color blotches. So, only during
// the session, everything outside a sphere around the subject is discarded in
// the shader (`crop`), on top of radial sorting and foveation (see `XrOptions`).

import {
  Application,
  Color,
  Entity,
  Quat,
  Vec3,
  type XrManager,
  XRHAND_RIGHT,
  XRPAD_STICK_X,
  XRPAD_STICK_Y,
  XRSPACE_LOCAL,
  XRSPACE_LOCALFLOOR,
  XRTYPE_AR,
  XRTYPE_VR,
} from "playcanvas";

export type XrMode = "vr" | "ar";

/** XR performance settings (defaults in main.ts, overridable from the URL). */
export interface XrOptions {
  /** XR framebuffer resolution (1 = the device's recommended one). */
  framebufferScale: number;
  /** Fixed foveation, 0–1: lower resolution at the lens edges. */
  foveation: number;
  /** Sorts splats by distance to the head rather than along the view direction:
   *  fewer artifacts when turning the head (PlayCanvas recommendation). */
  radialSort: boolean;
}

/** What becomes the diorama, in `stage` space. */
export interface XrContent {
  center: Vec3;
  /** Radius that maps to `STAGE_RADIUS`. */
  radius: number;
  /** Yaw (degrees, around +Y) that turns the filmed side to face the user. */
  yaw: number;
  /** Radius beyond which splats are hidden during the session; 0 disables cropping. */
  crop: number;
}

/** Diorama radius in XR (meters). */
const STAGE_RADIUS = 0.7;
/** Height of the scene center above the floor (`local-floor` spaces). */
const STAGE_CENTER_HEIGHT = 1.3;
/** Gap between the user and the nearest edge of the diorama (meters). */
const STAGE_CLEARANCE = 0.8;
/** Thumbstick and up/down speed (m/s, at human scale). */
const MOVE_SPEED = 1.5;
const LIFT_SPEED = 1.5;
const MOVE_DEADZONE = 0.15;
/** Snap turn: fires at the engage threshold, re-arms below the release one. */
const SNAP_TURN_DEGREES = 45;
const SNAP_ENGAGE = 0.6;
const SNAP_RELEASE = 0.3;

export interface XrCallbacks {
  /** immersive-vr / immersive-ar availability changed (also fired at startup). */
  onAvailability(vr: boolean, ar: boolean): void;
  /** Session started (true) or ended (false). */
  onSessionChange(active: boolean): void;
}

export class XrSession {
  private readonly xr: XrManager | null;
  private content: XrContent | null = null;
  private splat: Entity | null = null;
  private snapArmed = true;
  private readonly savedClearColor = new Color();
  private readonly detach: (() => void)[] = [];

  constructor(
    private readonly app: Application,
    private readonly cameraEntity: Entity,
    private readonly rig: Entity,
    private readonly stage: Entity,
    private readonly options: XrOptions,
    callbacks: XrCallbacks,
  ) {
    const xr = (this.xr = app.xr);
    if (!xr) {
      callbacks.onAvailability(false, false);
      return;
    }
    const emit = () =>
      callbacks.onAvailability(xr.supported && xr.isAvailable(XRTYPE_VR), xr.supported && xr.isAvailable(XRTYPE_AR));
    const handles = [
      xr.on("available", emit),
      xr.on("start", () => {
        this.onStart();
        callbacks.onSessionChange(true);
      }),
      xr.on("end", () => {
        this.onEnd();
        callbacks.onSessionChange(false);
      }),
      xr.on("error", (e: Error) => console.warn("[share] XR error", e)),
    ];
    this.detach.push(() => handles.forEach((h) => h.off()));
    emit();
  }

  get active() {
    return this.xr?.active ?? false;
  }

  /** Enters XR; must be called from a user gesture (click). Prefers
   *  `local-floor` (y = 0 on the real floor) and falls back to `local`. */
  start(mode: XrMode) {
    if (!this.xr?.supported || this.xr.active) return;
    const type = mode === "ar" ? XRTYPE_AR : XRTYPE_VR;
    this.startWith(type, XRSPACE_LOCALFLOOR, () => this.startWith(type, XRSPACE_LOCAL));
  }

  end() {
    if (this.xr?.active) this.xr.end();
  }

  /** What becomes the diorama (see `XrContent`). */
  setContent(content: XrContent) {
    this.content = { ...content, center: content.center.clone() };
    if (this.active) this.placeStage();
  }

  /** The splat entity, for cropping during the session. */
  setSplat(entity: Entity) {
    this.splat = entity;
    if (this.active) this.placeStage();
  }

  /** Builds the diorama outside XR, to check in the browser what the headset
   *  will show (`?xrPreview=1`). Returns the head position and the center. */
  preview() {
    this.app.scene.gsplat.radialSorting = this.options.radialSort;
    this.placeStage(STAGE_CENTER_HEIGHT);
    return { head: new Vec3(0, 1.6, 0), target: new Vec3(0, STAGE_CENTER_HEIGHT, -(STAGE_RADIUS + STAGE_CLEARANCE)) };
  }

  /** Move and turn with the controllers; call every frame during the session. */
  update(dt: number) {
    if (!this.xr?.active || dt <= 0) return;
    let moveX = 0;
    let moveZ = 0;
    let turn = 0;
    let lift = 0;
    for (const source of this.xr.input.inputSources) {
      const axes = source.gamepad?.axes;
      if (!axes?.length) continue;
      // The standard mapping puts the thumbstick at [2,3]; touchpad-only devices use [0,1].
      const x = axes[XRPAD_STICK_X] ?? axes[0] ?? 0;
      const y = axes[XRPAD_STICK_Y] ?? axes[1] ?? 0;
      if (source.handedness === XRHAND_RIGHT) {
        turn = x;
        lift = y;
      } else {
        moveX = x;
        moveZ = y;
      }
    }
    this.snapTurn(turn);
    this.glide(moveX, moveZ, lift, dt);
  }

  dispose() {
    this.end();
    this.detach.forEach((off) => off());
    this.detach.length = 0;
  }

  private startWith(type: string, space: string, onError?: () => void) {
    const camera = this.cameraEntity.camera;
    if (!camera || !this.xr) return;
    this.xr.start(camera, type, space, {
      framebufferScaleFactor: this.options.framebufferScale,
      callback: (err) => {
        if (!err) return;
        if (onError) onError();
        else console.warn("[share] could not enter XR", err);
      },
    });
  }

  private onStart() {
    // AR composites over the real world: transparent background during the session.
    const camera = this.cameraEntity.camera;
    if (camera && this.xr?.type === XRTYPE_AR) {
      this.savedClearColor.copy(camera.clearColor);
      camera.clearColor = new Color(0, 0, 0, 0);
    }
    this.snapArmed = true;
    this.app.scene.gsplat.radialSorting = this.options.radialSort;
    // null: the device has no fixed foveation.
    if (this.xr!.fixedFoveation !== null) this.xr!.fixedFoveation = this.options.foveation;
    this.placeStage();
  }

  private onEnd() {
    const camera = this.cameraEntity.camera;
    if (camera && this.xr?.type === XRTYPE_AR) camera.clearColor = this.savedClearColor.clone();
    this.app.scene.gsplat.radialSorting = false;
    this.splat?.gsplat?.setWorkBufferModifier(null);
    this.stage.setLocalScale(1, 1, 1);
    this.stage.setLocalPosition(0, 0, 0);
    this.stage.setLocalEulerAngles(0, 0, 0);
    this.rig.setLocalPosition(0, 0, 0);
    this.rig.setLocalEulerAngles(0, 0, 0);
  }

  /** Scales the scene into the diorama and stands it in front of the initial
   *  pose: radius → `STAGE_RADIUS`, center at chest height (with a floor) and
   *  `STAGE_CLEARANCE` beyond arm's reach, along -Z, with the filmed side
   *  facing the user. */
  private placeStage(height = this.xr?.spaceType === XRSPACE_LOCALFLOOR ? STAGE_CENTER_HEIGHT : 0) {
    const c = this.content;
    if (!c) return;
    const scale = STAGE_RADIUS / (c.radius || 1);
    const target = new Vec3(0, height, -(STAGE_RADIUS + STAGE_CLEARANCE));
    const rotation = new Quat().setFromEulerAngles(0, c.yaw, 0);
    // The content center must land on `target`: pos = target − R·(c·s).
    const offset = rotation.transformVector(c.center.clone().mulScalar(scale));
    this.stage.setLocalScale(scale, scale, scale);
    this.stage.setLocalEulerAngles(0, c.yaw, 0);
    this.stage.setLocalPosition(target.clone().sub(offset));
    // Cropping runs on PlayCanvas' work buffer, already in world coordinates:
    // the subject's center is `target` and the radius is scaled.
    if (this.splat?.gsplat && c.crop > 0) this.splat.gsplat.setWorkBufferModifier(cropShader(target, c.crop * scale));
  }

  /** Stepped rotation around the head: turning without motion sickness. */
  private snapTurn(turn: number) {
    const m = Math.abs(turn);
    if (m < SNAP_RELEASE) {
      this.snapArmed = true;
      return;
    }
    if (!this.snapArmed || m < SNAP_ENGAGE) return;
    this.snapArmed = false;
    const degrees = -Math.sign(turn) * SNAP_TURN_DEGREES;
    // Rotates the rig around the head position so the head stays in place.
    const head = this.cameraEntity.getPosition().clone();
    const offset = this.rig.getPosition().clone().sub(head);
    new Quat().setFromEulerAngles(0, degrees, 0).transformVector(offset, offset);
    this.rig.setPosition(offset.add(head));
    this.rig.rotate(0, degrees, 0);
  }

  /** Glides on the plane relative to the head; the right stick's Y moves up/down. */
  private glide(moveX: number, moveZ: number, lift: number, dt: number) {
    const moving = Math.hypot(moveX, moveZ) >= MOVE_DEADZONE;
    if (!moving && Math.abs(lift) < MOVE_DEADZONE) return;
    const forward = flat(this.cameraEntity.forward);
    const right = flat(this.cameraEntity.right);
    const step = MOVE_SPEED * dt;
    const vertical = Math.abs(lift) >= MOVE_DEADZONE ? -lift * LIFT_SPEED * dt : 0;
    const horizontal = moving && forward && right;
    // Stick up (negative Y) moves forward.
    this.rig.translate(
      horizontal ? (right.x * moveX - forward.x * moveZ) * step : 0,
      vertical,
      horizontal ? (right.z * moveX - forward.z * moveZ) * step : 0,
    );
  }
}

/** Discards (zero scale and opacity) the splats outside the sphere. */
function cropShader(center: Vec3, radius: number) {
  const f = (x: number) => x.toFixed(6);
  const c = `${f(center.x)}, ${f(center.y)}, ${f(center.z)}`;
  const r2 = f(radius * radius);
  return {
    glsl: `
      const vec3 gfCropCenter = vec3(${c});
      const float gfCropRadius2 = ${r2};
      bool gfOutside(vec3 p) { vec3 d = p - gfCropCenter; return dot(d, d) > gfCropRadius2; }
      void modifySplatCenter(inout vec3 center) {}
      void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
        if (gfOutside(originalCenter)) scale = vec3(0.0);
      }
      void modifySplatColor(vec3 center, inout vec4 color) {
        if (gfOutside(center)) color.a = 0.0;
      }`,
    wgsl: `
      const gfCropCenter = vec3f(${c});
      const gfCropRadius2 = ${r2};
      fn gfOutside(p: vec3f) -> bool { let d = p - gfCropCenter; return dot(d, d) > gfCropRadius2; }
      fn modifySplatCenter(center: ptr<function, vec3f>) {}
      fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
        if (gfOutside(originalCenter)) { *scale = vec3f(0.0); }
      }
      fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
        if (gfOutside(center)) { (*color).w = 0.0; }
      }`,
  };
}

/** Projects the direction onto the XZ plane and normalizes it (null if vertical). */
function flat(dir: Vec3): Vec3 | null {
  const v = new Vec3(dir.x, 0, dir.z);
  const l = v.length();
  return l > 1e-4 ? v.mulScalar(1 / l) : null;
}
