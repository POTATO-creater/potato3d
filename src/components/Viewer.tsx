import { useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import { fileUrl, type SceneFrame } from "../lib/api";
import { polarRange } from "../lib/share";
import { useI18n } from "../i18n";

interface Props {
  /** `label` overrides the default label (e.g. exported file); `url`, the
   *  address derived from `path` (e.g. a blob generated in the browser). */
  splat: { path: string; iter: number | null; version: number; label?: string; url?: string } | null;
  frame: SceneFrame | null;
  placeholder?: React.ReactNode;
  /** Click on a trajectory point: camera index in `frame.cameras`. */
  onPickCamera?: (camera: number) => void;
  ref?: Ref<ViewerHandle>;
}

/** Pose in COLMAP convention (world → camera). */
export interface ColmapPose {
  qvec: [number, number, number, number];
  tvec: [number, number, number];
}

/** Retouch video frame: aspect and vertical field of view (rad). */
export interface FrameGuide {
  size: [number, number];
  fovY: number;
}

/** Retouch access to the viewer: the current camera, the drawn trajectory
 *  and offscreen rendering, in COLMAP coordinates. */
export interface ViewerHandle {
  /** Current camera; null before framing. */
  viewPose(): ColmapPose | null;
  /** Draws a retouch trajectory (null clears it). With `frame` (video size and
   *  vertical focal length), also draws the camera at a few points. */
  showPath(poses: ColmapPose[] | null, frame?: { size: [number, number]; fy: number }): void;
  /** Moves the camera to a pose, looking at the anchor (or the orbit center). */
  goTo(pose: ColmapPose): void;
  /** Releases the angle lock (aim below or above the filmed range). */
  unlockAngle(): void;
  /** A splat is loaded on screen. */
  hasSplat(): boolean;
  /** A mouse button (or finger) is pressed on the 3D view. */
  pointerDown(): boolean;
  /** Orbit pivot placed with the Anchor gizmo, in COLMAP coordinates; null
   *  until it is moved (and after Recenter). */
  anchorPoint(): [number, number, number] | null;
  /** Video frame for the framing mask (null removes the option). */
  setFrameGuide(guide: FrameGuide | null): void;
  /** Renders each pose offscreen with the displayed splat. `onFrame` gets
   *  top-down RGBA with the color already un-premultiplied. */
  renderFrames(opts: {
    poses: ColmapPose[];
    size: [number, number];
    fy: number;
    onFrame: (index: number, rgba: Uint8ClampedArray) => void | Promise<void>;
    signal?: AbortSignal;
  }): Promise<void>;
}

interface Ctx {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  /** Holds the COLMAP → world transform (center at origin, "up" on +Y). */
  root: THREE.Group;
  /** The same transform, but from the poses the displayed splat was trained
   *  with: when poses are redone, the old splat stays on screen until the
   *  new training's first snapshot and must not inherit the new alignment
   *  (nor its absence, which would leave it upside down). */
  splatRoot: THREE.Group;
  /** `splatRoot` already has the displayed splat's alignment. */
  splatAligned: boolean;
  mesh: SplatMesh | null;
  /** An exported mesh (.glb), shown instead of a splat: already y-up and
   *  centred, so outside `splatRoot`, with its own lights. */
  glb: THREE.Group | null;
  path: THREE.Object3D | null;
  /** Highlight of the point under the cursor (child of `path`). */
  hover: THREE.Points | null;
  /** Retouch trajectory (child of `root`). */
  retouchPath: THREE.Object3D | null;
  /** Requests a few render frames (the scene only redraws on change). */
  invalidate: () => void;
  /** Pointers pressed on the canvas (released anywhere). */
  pressed: Set<number>;
  /** Orbit pivot (`controls.target`) shown and moved with a gizmo. */
  anchor: THREE.Object3D;
  anchorGizmo: TransformControls;
  /** Where the pivot is gliding to after a gizmo drag (null = at rest). */
  anchorGoal: THREE.Vector3 | null;
  /** The pivot was placed with the gizmo (it then follows pans too). */
  anchored: boolean;
}

// Extra frames after each change: cover orbit damping and Spark's async
// splat sorting.
const SETTLE_FRAMES = 45;

const ACCENT = 0xff9a4d;
/** Radius, in px, for hitting a camera point with the mouse. */
const PICK_RADIUS = 10;
/** Fraction of the pending wheel zoom applied per frame. */
const ZOOM_EASE = 0.2;
/** Fraction of the way to its new spot the pivot moves per frame. */
const ANCHOR_EASE = 0.15;
/** Default viewer vertical field of view, in degrees. */
const BASE_FOV = 50;
/** Maximum fraction of the screen the framing box takes. */
const GUIDE_FILL = 0.86;

/** Frees the geometries, materials and textures under `o`. */
function disposeTree(o: THREE.Object3D) {
  o.traverse((x) => {
    const m = x as THREE.Mesh;
    m.geometry?.dispose();
    for (const mat of ([] as THREE.Material[]).concat(m.material ?? [])) {
      for (const v of Object.values(mat)) if (v instanceof THREE.Texture) v.dispose();
      mat.dispose();
    }
  });
}

export function Viewer({ splat, frame, placeholder, onPickCamera, ref }: Props) {
  const { t, locale } = useI18n();
  const tv = t.viewer;
  const host = useRef<HTMLDivElement>(null);
  const ctx = useRef<Ctx | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showCams, setShowCams] = useState(true);
  const [spin, setSpin] = useState(false);
  const [clampView, setClampView] = useState(false);
  const [anchorOn, setAnchorOn] = useState(false);
  const [hasMesh, setHasMesh] = useState(false);
  /** An exported .glb is on screen (the training cameras don't line up with it). */
  const [glbShown, setGlbShown] = useState(false);
  const [guide, setGuide] = useState<FrameGuide | null>(null);
  const [guideOn, setGuideOn] = useState(false);
  /** On-screen size in px of the frame box, with the mask on. */
  const [guideBox, setGuideBox] = useState<{ w: number; h: number } | null>(null);
  const activeGuide = useRef<FrameGuide | null>(null);
  const fitGuideRef = useRef<() => void>(() => {});
  // Canvas listeners are created once; they read current state through these.
  const frameRef = useRef(frame);
  frameRef.current = frame;
  const pickRef = useRef(onPickCamera);
  pickRef.current = onPickCamera;

  // Scene, renderer and loop: created once.
  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    camera.position.set(0, 1, 4);
    const spark = new SparkRenderer({ renderer });
    scene.add(spark);
    const root = new THREE.Group();
    const splatRoot = new THREE.Group();
    scene.add(root, splatRoot);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotateSpeed = 1.2;

    // Rendering only on demand frees the GPU for Brush training, which runs
    // in parallel on the same chip.
    let dirty = SETTLE_FRAMES;
    const invalidate = () => {
      dirty = SETTLE_FRAMES;
    };

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = el;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      fitGuideRef.current();
      invalidate();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    // Smooth wheel zoom. OrbitControls applies each wheel notch at once (a jump
    // in a single frame); here the zoom glides like the rotation damping, so a
    // recorded retouch motion zooms continuously instead of in steps. Listens
    // in the capture phase on the container so OrbitControls never sees it.
    let zoomPending = 0; // log of the distance factor still to apply
    const onWheel = (e: WheelEvent) => {
      if (!controls.enabled || !controls.enableZoom) return;
      e.preventDefault();
      e.stopPropagation();
      // Same scale as OrbitControls: 5% per 100 px, lines and pages in px,
      // trackpad pinch (ctrl) 10× stronger.
      let dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1);
      if (e.ctrlKey) dy *= 10;
      zoomPending += (dy / 100) * -Math.log(0.95) * controls.zoomSpeed;
      invalidate();
    };
    el.addEventListener("wheel", onWheel, { capture: true, passive: false });
    const offset = new THREE.Vector3();
    const stepZoom = () => {
      const step = Math.abs(zoomPending) < 1e-4 ? zoomPending : zoomPending * ZOOM_EASE;
      zoomPending -= step;
      offset.copy(camera.position).sub(controls.target);
      const d = THREE.MathUtils.clamp(offset.length() * Math.exp(step), controls.minDistance, controls.maxDistance);
      camera.position.copy(controls.target).add(offset.setLength(d));
    };

    // Orbit pivot: a point with a translate gizmo. While the gizmo is dragged
    // the camera stays put (if it kept aiming at the pivot, the gizmo would
    // stay pinned to the screen center); on release the pivot glides to the
    // new spot and the camera turns to aim at it.
    const anchor = new THREE.Object3D();
    anchor.add(
      new THREE.Points(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3()]),
        new THREE.PointsMaterial({ color: 0xffffff, size: 8, sizeAttenuation: false, depthTest: false, transparent: true }),
      ),
    );
    anchor.children[0].renderOrder = 10;
    anchor.visible = false;
    scene.add(anchor);
    const gizmo = new TransformControls(camera, renderer.domElement);
    gizmo.setMode("translate");
    gizmo.setSize(0.8);
    gizmo.enabled = false;
    gizmo.attach(anchor);
    const gizmoHelper = gizmo.getHelper();
    gizmoHelper.visible = false;
    scene.add(gizmoHelper);
    gizmo.addEventListener("change", invalidate);
    gizmo.addEventListener("dragging-changed", (e) => {
      controls.enabled = !e.value;
      if (!e.value && ctx.current) {
        ctx.current.anchorGoal = anchor.position.clone();
        ctx.current.anchored = true;
      }
    });
    const stepAnchor = () => {
      const c = ctx.current!;
      if (gizmo.dragging) return;
      if (c.anchorGoal) {
        controls.target.lerp(c.anchorGoal, ANCHOR_EASE);
        if (controls.target.distanceToSquared(c.anchorGoal) < 1e-10) {
          controls.target.copy(c.anchorGoal);
          c.anchorGoal = null;
        }
        dirty = SETTLE_FRAMES;
      } else if (!anchor.position.equals(controls.target)) {
        // Panning and recentering move the pivot: the gizmo follows.
        anchor.position.copy(controls.target);
        dirty = SETTLE_FRAMES;
      }
    };

    controls.addEventListener("change", invalidate);
    renderer.setAnimationLoop(() => {
      if (zoomPending) stepZoom();
      if (ctx.current) stepAnchor();
      if (controls.update() || controls.autoRotate) dirty = SETTLE_FRAMES;
      if (dirty <= 0) return;
      dirty--;
      renderer.render(scene, camera);
    });

    ctx.current = { renderer, scene, camera, controls, root, splatRoot, splatAligned: false, mesh: null, glb: null, path: null, hover: null, retouchPath: null, invalidate, pressed: new Set(), anchor, anchorGizmo: gizmo, anchorGoal: null, anchored: false };

    // Camera point closest to the cursor, in screen pixels.
    const v = new THREE.Vector3();
    const pick = (e: PointerEvent) => {
      const c = ctx.current;
      const f = frameRef.current;
      if (!c?.path?.visible || !f || !pickRef.current) return -1;
      const rect = renderer.domElement.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      root.updateMatrixWorld(true);
      let best = -1;
      let bestD = PICK_RADIUS;
      f.cameras.forEach((p, i) => {
        v.set(...p).applyMatrix4(root.matrixWorld).project(camera);
        if (v.z < -1 || v.z > 1) return;
        const d = Math.hypot(((v.x + 1) / 2) * rect.width - px, ((1 - v.y) / 2) * rect.height - py);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      });
      return best;
    };
    let hovered = -1;
    const setHover = (i: number) => {
      const c = ctx.current;
      // After a trajectory change the new highlight starts hidden: redo it even
      // if the index is the same.
      if (i === hovered && (i < 0 || c?.hover?.visible)) return;
      hovered = i;
      const f = frameRef.current;
      renderer.domElement.style.cursor = i >= 0 ? "pointer" : "";
      if (!c?.hover) return;
      c.hover.visible = i >= 0 && !!f;
      if (i >= 0 && f) c.hover.position.set(...f.cameras[i]);
      invalidate();
    };
    // Click = release near the same spot; dragging still orbits.
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => {
      ctx.current?.pressed.add(e.pointerId);
      down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
    };
    // The release may happen outside the canvas.
    const onRelease = (e: PointerEvent) => ctx.current?.pressed.delete(e.pointerId);
    const onUp = (e: PointerEvent) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      down = null;
      const i = pick(e);
      if (i >= 0) pickRef.current?.(i);
    };
    const onMove = (e: PointerEvent) => setHover(e.buttons ? -1 : pick(e));
    const onLeave = () => setHover(-1);
    const canvas = renderer.domElement;
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    window.addEventListener("pointerup", onRelease);
    window.addEventListener("pointercancel", onRelease);

    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      window.removeEventListener("pointerup", onRelease);
      window.removeEventListener("pointercancel", onRelease);
      el.removeEventListener("wheel", onWheel, { capture: true });
      ro.disconnect();
      renderer.setAnimationLoop(null);
      gizmo.detach();
      gizmo.dispose();
      controls.dispose();
      ctx.current?.mesh?.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      ctx.current = null;
    };
  }, []);

  // Framing from the COLMAP cameras.
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    if (c.path) {
      c.root.remove(c.path);
      c.path = null;
      c.hover = null;
    }
    alignTo(c.root, frame);
    if (!c.splatAligned) alignSplat(frame);
    if (!frame) return;

    const pts = frame.cameras.map((p) => new THREE.Vector3(...p));
    const group = new THREE.Group();
    if (pts.length > 1) {
      // Only connect adjacent frames: an index jump is a stretch COLMAP could
      // not register and should show up as a gap in the trajectory.
      const segs: THREE.Vector3[] = [];
      for (let i = 1; i < pts.length; i++) {
        if (frame.frameIndices[i] - frame.frameIndices[i - 1] === 1) segs.push(pts[i - 1], pts[i]);
      }
      const line = new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints(segs),
        new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.45 }),
      );
      const dots = new THREE.Points(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.PointsMaterial({ color: ACCENT, size: 4, sizeAttenuation: false }),
      );
      const hover = new THREE.Points(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3()]),
        new THREE.PointsMaterial({ color: 0xffffff, size: 9, sizeAttenuation: false, depthTest: false }),
      );
      hover.visible = false;
      hover.renderOrder = 1;
      group.add(line, dots, hover);
      c.hover = hover;
    }
    group.visible = showCams && !c.glb;
    c.root.add(group);
    c.invalidate();
    c.path = group;
    resetView();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  useEffect(() => {
    if (ctx.current?.path) ctx.current.path.visible = showCams && !glbShown;
    ctx.current?.invalidate();
  }, [showCams, glbShown]);

  useEffect(() => {
    if (ctx.current) ctx.current.controls.autoRotate = spin;
  }, [spin]);

  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    c.anchorGizmo.enabled = anchorOn;
    c.anchorGizmo.getHelper().visible = anchorOn;
    c.anchor.visible = anchorOn;
    c.invalidate();
  }, [anchorOn]);

  // Framing mask: darkens whatever is outside the retouch video frame. The
  // video lens may be wider than the viewer's; then the viewer widens its
  // field of view until the frame fits on screen.
  fitGuideRef.current = () => {
    const c = ctx.current;
    const el = host.current;
    const g = activeGuide.current;
    if (!c || !el) return;
    const { clientWidth: w, clientHeight: h } = el;
    let fov = BASE_FOV;
    let box: { w: number; h: number } | null = null;
    if (g && w && h) {
      const aspect = g.size[0] / g.size[1];
      const tc = Math.tan(g.fovY / 2);
      const tv = Math.max(Math.tan(THREE.MathUtils.degToRad(BASE_FOV) / 2), (tc * Math.max(1, aspect / (w / h))) / GUIDE_FILL);
      fov = THREE.MathUtils.radToDeg(2 * Math.atan(tv));
      const bh = (h * tc) / tv;
      box = { w: bh * aspect, h: bh };
    }
    if (Math.abs(c.camera.fov - fov) > 1e-6) {
      c.camera.fov = fov;
      c.camera.updateProjectionMatrix();
      c.invalidate();
    }
    setGuideBox((cur) => (cur?.w === box?.w && cur?.h === box?.h ? cur : box));
  };
  useEffect(() => {
    activeGuide.current = guideOn && guide ? guide : null;
    fitGuideRef.current();
  }, [guide, guideOn]);

  // Clamp orbit elevation to the range the video covered (+ margin): outside
  // it there is no data and the splat shows holes and artifacts.
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    const range = clampView && frame ? polarRange(frame) : null;
    c.controls.minPolarAngle = range ? range[0] : 0;
    c.controls.maxPolarAngle = range ? range[1] : Math.PI;
    c.controls.update();
    c.invalidate();
  }, [clampView, frame]);

  // Loads the .ply (snapshot or final). Swaps without flicker: the previous
  // one is removed only once the new one has loaded.
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    if (c.glb) {
      c.scene.remove(c.glb);
      disposeTree(c.glb);
      c.glb = null;
      c.invalidate();
      setGlbShown(false);
      if (!c.mesh) setHasMesh(false);
    }
    if (splat?.path.toLowerCase().endsWith(".glb")) {
      let stale = false;
      setLoading(true);
      setLoadError(null);
      new GLTFLoader()
        .loadAsync(splat.url ?? fileUrl(splat.path, splat.version))
        .then((gltf) => {
          if (stale || !ctx.current) return disposeTree(gltf.scene);
          // the splat (if any) steps aside while the mesh is shown
          if (c.mesh) c.mesh.visible = false;
          const g = new THREE.Group();
          const sun = new THREE.DirectionalLight(0xffffff, 1.6);
          sun.position.set(2, 4, 3);
          g.add(gltf.scene, new THREE.HemisphereLight(0xffffff, 0x404040, 1.4), sun);
          c.scene.add(g);
          c.glb = g;
          resetView();
          setGlbShown(true);
          setHasMesh(true);
        })
        .catch((e) => !stale && setLoadError(String(e)))
        .finally(() => !stale && setLoading(false));
      return () => {
        stale = true;
      };
    }
    if (c.mesh) c.mesh.visible = true;
    if (!splat) {
      if (c.mesh) {
        c.splatRoot.remove(c.mesh);
        c.mesh.dispose();
        c.mesh = null;
        c.invalidate();
        setHasMesh(false);
      }
      alignSplat(frameRef.current);
      return;
    }
    let stale = false;
    setLoading(true);
    setLoadError(null);
    const fileType = splat.path.toLowerCase().endsWith(".sog") ? SplatFileType.PCSOGSZIP : SplatFileType.PLY;
    const mesh = new SplatMesh({ url: splat.url ?? fileUrl(splat.path, splat.version), fileType });
    mesh.initialized
      .then(() => {
        if (stale || !ctx.current) {
          mesh.dispose();
          return;
        }
        const prev = c.mesh;
        if (prev) {
          c.splatRoot.remove(prev);
          prev.dispose();
        }
        // New splat: starts with the current poses' alignment.
        alignSplat(frameRef.current);
        c.splatRoot.add(mesh);
        c.invalidate();
        c.mesh = mesh;
        if (!frame && !prev) fitToMesh(mesh);
        setHasMesh(true);
      })
      .catch((e) => !stale && setLoadError(String(e)))
      .finally(() => !stale && setLoading(false));
    return () => {
      stale = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [splat?.path, splat?.version]);

  useImperativeHandle(
    ref,
    (): ViewerHandle => {
      const handle: ViewerHandle = {
      viewPose() {
        const c = ctx.current;
        if (!c || !frameRef.current) return null;
        c.root.updateMatrixWorld(true);
        c.camera.updateMatrixWorld(true);
        // three camera (looks −Z, +Y up) → COLMAP camera (looks +Z, +Y down)
        const c2w = c.root.matrixWorld.clone().invert().multiply(c.camera.matrixWorld).multiply(FLIP_YZ);
        return colmapFromC2W(c2w);
      },
      showPath(poses, frame) {
        const c = ctx.current;
        if (!c) return;
        if (c.retouchPath) {
          c.root.remove(c.retouchPath);
          c.retouchPath.traverse((o) => {
            (o as THREE.Mesh).geometry?.dispose();
            ((o as THREE.Mesh).material as THREE.Material | undefined)?.dispose();
          });
          c.retouchPath = null;
        }
        if (poses?.length) {
          const pts = poses.map((p) => centerOf(p));
          const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), pathMaterial(RETOUCH, 0.95));
          // Ticks every ~8% of the clip: show the camera's pace and direction.
          const step = Math.max(1, Math.round(pts.length / 12));
          const ticks = new THREE.Points(
            new THREE.BufferGeometry().setFromPoints(pts.filter((_, i) => i % step === 0)),
            pathMaterial(RETOUCH, 0.95, 5),
          );
          const end = new THREE.Points(new THREE.BufferGeometry().setFromPoints([pts[pts.length - 1]]), pathMaterial(RETOUCH, 1, 9));
          const mid = new THREE.Points(new THREE.BufferGeometry().setFromPoints([pts[pts.length >> 1]]), pathMaterial(0xffffff, 1, 10));
          // Start last, on top of the end (they coincide on full loops).
          const start = new THREE.Points(new THREE.BufferGeometry().setFromPoints([pts[0]]), pathMaterial(PATH_START, 1, 11));
          const g = new THREE.Group();
          g.add(line, ticks, end, mid, start);
          // Cameras at the ticks (and at start and middle): a pyramid whose base is
          // the frame shape, with a triangle on the image's top side.
          if (frame) {
            const depth = (frameRef.current?.radius ?? 1) * 0.07;
            const marks = poses.map((_, i) => i).filter((i) => i % step === 0 && i !== 0 && i !== poses.length >> 1);
            const frustums = (idx: number[], color: number, opacity: number) =>
              new THREE.LineSegments(
                new THREE.BufferGeometry().setFromPoints(idx.flatMap((i) => frustumSegments(poses[i], frame, depth))),
                pathMaterial(color, opacity),
              );
            g.add(frustums(marks, RETOUCH, 0.75), frustums([poses.length >> 1], 0xffffff, 0.95), frustums([0], PATH_START, 1));
          }
          g.children.forEach((o, i) => (o.renderOrder = 2 + i));
          c.root.add(g);
          c.retouchPath = g;
        }
        c.invalidate();
      },
      goTo(pose) {
        const c = ctx.current;
        if (!c) return;
        // The pose may be outside the filmed range: release the lock before moving
        // (the state effect only runs later, and update() would already clamp).
        setClampView(false);
        c.controls.minPolarAngle = 0;
        c.controls.maxPolarAngle = Math.PI;
        c.root.updateMatrixWorld(true);
        // Looks at the anchor when there is one, otherwise at the orbit center.
        if (c.anchorGoal) c.controls.target.copy(c.anchorGoal);
        c.anchorGoal = null;
        if (!c.anchored) c.controls.target.set(0, 0, 0);
        c.camera.position.copy(centerOf(pose).applyMatrix4(c.root.matrixWorld));
        c.controls.update();
        c.invalidate();
      },
      unlockAngle() {
        setClampView(false);
        if (ctx.current) {
          ctx.current.controls.minPolarAngle = 0;
          ctx.current.controls.maxPolarAngle = Math.PI;
        }
      },
      hasSplat() {
        return !!ctx.current?.mesh;
      },
      pointerDown() {
        return !!ctx.current?.pressed.size;
      },
      anchorPoint() {
        const c = ctx.current;
        if (!c?.anchored || !frameRef.current) return null;
        c.root.updateMatrixWorld(true);
        const p = (c.anchorGoal ?? c.controls.target).clone().applyMatrix4(c.root.matrixWorld.clone().invert());
        return [p.x, p.y, p.z];
      },
      setFrameGuide(g) {
        setGuide(g);
      },
      async renderFrames({ poses, size: [w, h], fy, onFrame, signal }) {
        const c = ctx.current;
        if (!c?.mesh) throw new Error("no splat loaded in the viewer");
        const f = frameRef.current;
        // RGBA8 target without sRGB encoding: values read back are exactly what
        // the shader wrote (see the render below).
        const spark = new SparkRenderer({ renderer: c.renderer, target: { width: w, height: h, colorSpace: THREE.NoColorSpace }, minSortIntervalMs: 0 });
        const cam = new THREE.PerspectiveCamera(THREE.MathUtils.radToDeg(2 * Math.atan(h / 2 / fy)), w / h, 0.01, 1000);
        if (f) {
          cam.near = Math.max(f.radius / 500, 0.001);
          cam.far = f.radius * 100;
        }
        cam.updateProjectionMatrix();
        cam.matrixAutoUpdate = false;
        // Camera and trajectory lines must not show up in the video.
        const hidden = [c.path, c.retouchPath, c.anchor, c.anchorGizmo.getHelper()].filter((o): o is THREE.Object3D => !!o && o.visible);
        hidden.forEach((o) => (o.visible = false));
        const row = w * 4;
        const out = new Uint8ClampedArray(w * h * 4);
        try {
          c.root.updateMatrixWorld(true);
          for (let i = 0; i < poses.length; i++) {
            if (signal?.aborted) throw new DOMException("cancelled", "AbortError");
            cam.matrix.copy(c.root.matrixWorld).multiply(c2wFromColmap(poses[i])).multiply(FLIP_YZ);
            cam.matrixWorld.copy(cam.matrix);
            cam.matrixWorldInverse.copy(cam.matrix).invert();
            // Splat sorting is async: wait for this pose's sort.
            await spark.update({ scene: c.scene, camera: cam });
            while (spark.sorting || spark.sortDirty) {
              await new Promise((r) => setTimeout(r, 2));
              if (spark.sortDirty && !spark.sorting) await spark.update({ scene: c.scene, camera: cam });
            }
            // In a render target Spark converts colors to linear and blends splats
            // in linear space (encodeLinear, driven by three's working color
            // space). On screen, and in Brush which trained the splat, blending
            // is in sRGB; in linear, darks wash out and highlights flatten.
            // During the (synchronous) render the working space is switched to
            // sRGB so Spark writes colors as on screen.
            const working = THREE.ColorManagement.workingColorSpace;
            THREE.ColorManagement.workingColorSpace = THREE.SRGBColorSpace;
            try {
              spark.renderTarget({ scene: c.scene, camera: cam });
            } finally {
              THREE.ColorManagement.workingColorSpace = working;
            }
            const px = await spark.readTarget();
            // WebGL reads bottom-up; the video wants top-down.
            for (let y = 0; y < h; y++) out.set(px.subarray((h - 1 - y) * row, (h - y) * row), y * row);
            if (spark.premultipliedAlpha) {
              for (let k = 0; k < out.length; k += 4) {
                const a = out[k + 3];
                if (a > 0 && a < 255) {
                  out[k] = (out[k] * 255) / a;
                  out[k + 1] = (out[k + 1] * 255) / a;
                  out[k + 2] = (out[k + 2] * 255) / a;
                }
              }
            }
            await onFrame(i, out);
          }
        } finally {
          hidden.forEach((o) => (o.visible = true));
          spark.dispose();
          c.invalidate();
        }
      },
      };
      // For automated tests (scripts/web-bench/retouch.mjs).
      if (import.meta.env.DEV) (window as unknown as { __gorbitViewer?: ViewerHandle }).__gorbitViewer = handle;
      return handle;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  /** Aligns the splat to `frame`; with no frame, waits for the next one. */
  function alignSplat(frame: SceneFrame | null) {
    const c = ctx.current!;
    alignTo(c.splatRoot, frame);
    c.splatAligned = !!frame;
    c.invalidate();
  }

  function fitToMesh(mesh: SplatMesh) {
    const c = ctx.current!;
    c.splatRoot.updateMatrixWorld(true);
    fitToBox(mesh.getBoundingBox(true).applyMatrix4(c.splatRoot.matrixWorld));
  }

  /** Frames `box` from the front: `back` and `lift` in box diagonals. */
  function fitToBox(box: THREE.Box3, back = 0.9, lift = 0.25) {
    const c = ctx.current!;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    c.controls.target.copy(center);
    c.camera.position.copy(center).add(new THREE.Vector3(0, size * lift, size * back));
  }

  /** Returns to the viewpoint of the video's first frame, framing the center
   *  like the footage (or slightly pulled back, without the video lens). */
  function resetView() {
    const c = ctx.current;
    if (!c) return;
    c.anchorGoal = null;
    c.anchored = false;
    c.controls.target.set(0, 0, 0);
    if (c.glb) {
      // a tight box (no floaters, unlike splats): step back to see it whole
      fitToBox(new THREE.Box3().setFromObject(c.glb.children[0]), 1.3, 0.1);
    } else if (frame && frame.cameras.length) {
      c.root.updateMatrixWorld(true);
      const first = new THREE.Vector3(...frame.cameras[0]).applyMatrix4(c.root.matrixWorld);
      // The viewer lens is fixed (50°); the video's may be much narrower
      // (telephoto). Move in until the video frame fits the screen snugly.
      let k = 1.35;
      if (frame.lens) {
        const v = Math.tan(THREE.MathUtils.degToRad(c.camera.fov) / 2);
        const tv = Math.tan(frame.lens.fovY / 2);
        k = Math.max(tv / v, (tv * frame.lens.aspect) / (v * c.camera.aspect));
      }
      c.camera.position.copy(first.multiplyScalar(k));
      c.camera.near = Math.max(frame.radius / 500, 0.001);
      c.camera.far = frame.radius * 100;
      c.camera.updateProjectionMatrix();
    } else if (c.mesh) {
      fitToMesh(c.mesh);
    }
    c.controls.update();
    c.invalidate();
  }

  return (
    <div className="viewer">
      <div className="viewer-canvas" ref={host} />
      {!hasMesh && !loading && <div className="viewer-empty">{placeholder}</div>}
      {guide && guideBox && (
        <div className="viewer-guide" style={{ width: guideBox.w, height: guideBox.h }} aria-hidden>
          <span>{tv.guideLabel(guide.size[0], guide.size[1])}</span>
        </div>
      )}
      <div className="viewer-hud">
        {splat && (
          <span className="chip">
            {splat.label ?? (splat.iter != null ? tv.preview(splat.iter.toLocaleString(locale)) : tv.final)}
          </span>
        )}
        {loading && <span className="chip chip-pulse">{tv.loading}</span>}
        {loadError && <span className="chip chip-error">{tv.loadError(loadError)}</span>}
      </div>
      <div className="viewer-tools">
        <button
          className={`tool ${showCams ? "on" : ""}`}
          onClick={() => setShowCams((v) => !v)}
          disabled={!frame}
          title={onPickCamera ? tv.camerasTitle : undefined}
        >
          {tv.cameras}
        </button>
        <button
          className={`tool ${clampView ? "on" : ""}`}
          onClick={() => setClampView((v) => !v)}
          disabled={!frame}
          title={tv.lockAngleTitle}
        >
          {tv.lockAngle}
        </button>
        <button className={`tool ${anchorOn ? "on" : ""}`} onClick={() => setAnchorOn((v) => !v)} disabled={!hasMesh} title={tv.anchorTitle}>
          {tv.anchor}
        </button>
        {guide && (
          <button className={`tool ${guideOn ? "on" : ""}`} onClick={() => setGuideOn((v) => !v)} title={tv.guideTitle}>
            {tv.guide}
          </button>
        )}
        <button className={`tool ${spin ? "on" : ""}`} onClick={() => setSpin((v) => !v)}>
          {tv.spin}
        </button>
        <button className="tool" onClick={resetView}>
          {tv.recenter}
        </button>
      </div>
    </div>
  );
}

const RETOUCH = 0x7fd3ff;
/** Start of the retouch trajectory. */
const PATH_START = 0x6fe3a1;

/** Retouch trajectory material: drawn over the splat (which writes no
 *  depth), but faded where it passes behind the orbit center (the world
 *  origin). Without this a loop around the subject, seen almost edge-on,
 *  looks like a diagonal crossing the body. `size` > 0: points. */
function pathMaterial(color: number, opacity: number, size = 0) {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(color) }, uOpacity: { value: opacity }, uSize: { value: size } },
    vertexShader: /* glsl */ `
      uniform float uSize;
      varying float vBehind;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float cz = (viewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).z;
        vBehind = smoothstep(-0.04, 0.04, (cz - mv.z) / max(-cz, 1e-3));
        gl_PointSize = uSize;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uSize;
      varying float vBehind;
      void main() {
        if (uSize > 0.0 && length(gl_PointCoord - 0.5) > 0.5) discard;
        gl_FragColor = vec4(uColor, uOpacity * mix(1.0, 0.22, vBehind));
      }`,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
}
/** Converts between three camera axes (+Y up, looks −Z) and COLMAP's. */
const FLIP_YZ = new THREE.Matrix4().makeScale(1, -1, -1);

/** Camera → world (COLMAP) from a world → camera pose. */
function c2wFromColmap({ qvec: [w, x, y, z], tvec }: ColmapPose) {
  const r = new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(x, y, z, w));
  const t = new THREE.Matrix4().makeTranslation(tvec[0], tvec[1], tvec[2]);
  return t.multiply(r).invert();
}

const centerOf = (p: ColmapPose) => new THREE.Vector3().setFromMatrixPosition(c2wFromColmap(p));

/** Segments (point pairs) of a drawn camera: a pyramid from the position to
 *  the frame at `depth` ahead, the frame itself and a triangle on the image's
 *  top side. COLMAP coordinates (+Z forward, +Y down). */
function frustumSegments(p: ColmapPose, frame: { size: [number, number]; fy: number }, depth: number): THREE.Vector3[] {
  const c2w = c2wFromColmap(p);
  const hh = (depth * frame.size[1]) / 2 / frame.fy;
  const hw = (hh * frame.size[0]) / frame.size[1];
  const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z).applyMatrix4(c2w);
  const o = v(0, 0, 0);
  const [tl, tr, br, bl] = [v(-hw, -hh, depth), v(hw, -hh, depth), v(hw, hh, depth), v(-hw, hh, depth)];
  const [ul, ur, ut] = [v(-hw * 0.35, -hh * 1.08, depth), v(hw * 0.35, -hh * 1.08, depth), v(0, -hh * 1.45, depth)];
  return [o, tl, o, tr, o, br, o, bl, tl, tr, tr, br, br, bl, bl, tl, ul, ut, ut, ur, ur, ul];
}

function colmapFromC2W(c2w: THREE.Matrix4): ColmapPose {
  const w2c = c2w.clone().invert();
  const q = new THREE.Quaternion().setFromRotationMatrix(w2c);
  const t = new THREE.Vector3().setFromMatrixPosition(w2c);
  const sign = q.w < 0 ? -1 : 1;
  return { qvec: [q.w * sign, q.x * sign, q.y * sign, q.z * sign], tvec: [t.x, t.y, t.z] };
}

/** COLMAP → world transform: center at origin, "up" on +Y. */
function alignTo(group: THREE.Group, frame: SceneFrame | null) {
  if (!frame) {
    group.position.set(0, 0, 0);
    group.quaternion.identity();
    return;
  }
  const up = new THREE.Vector3(...frame.up).normalize();
  const q = new THREE.Quaternion().setFromUnitVectors(up, new THREE.Vector3(0, 1, 0));
  const center = new THREE.Vector3(...frame.center).applyQuaternion(q);
  group.quaternion.copy(q);
  group.position.copy(center.negate());
}
