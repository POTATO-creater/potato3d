import { useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import { fileUrl, type SceneFrame } from "../lib/api";
import { polarRange } from "../lib/share";
import { useI18n } from "../i18n";

interface Props {
  /** `label` substitui o rótulo padrão (ex.: arquivo exportado); `url`, o
   *  endereço derivado de `path` (ex.: blob gerado no navegador). */
  splat: { path: string; iter: number | null; version: number; label?: string; url?: string } | null;
  frame: SceneFrame | null;
  placeholder?: React.ReactNode;
  /** Clique num ponto da trajetória: índice da câmera em `frame.cameras`. */
  onPickCamera?: (camera: number) => void;
  ref?: Ref<ViewerHandle>;
}

/** Pose em convenção COLMAP (mundo → câmera). */
export interface ColmapPose {
  qvec: [number, number, number, number];
  tvec: [number, number, number];
}

/** Quadro do vídeo de retoque: proporção e campo de visão vertical (rad). */
export interface FrameGuide {
  size: [number, number];
  fovY: number;
}

/** Acesso do retoque ao viewer: a câmera atual, a trajetória desenhada e o
 *  render fora da tela, nas coordenadas do COLMAP. */
export interface ViewerHandle {
  /** Câmera atual; null antes do enquadramento. */
  viewPose(): ColmapPose | null;
  /** Desenha uma trajetória de retoque (null apaga). Com `frame` (tamanho e
   *  focal vertical do vídeo), desenha também a câmera em alguns pontos. */
  showPath(poses: ColmapPose[] | null, frame?: { size: [number, number]; fy: number }): void;
  /** Leva a câmera a uma pose, olhando para o centro da órbita. */
  goTo(pose: ColmapPose): void;
  /** Solta a trava de ângulo (mirar por baixo ou por cima da filmagem). */
  unlockAngle(): void;
  /** Splat carregado na tela. */
  hasSplat(): boolean;
  /** Quadro do vídeo para a máscara de enquadramento (null tira a opção). */
  setFrameGuide(guide: FrameGuide | null): void;
  /** Renderiza cada pose fora da tela, com o splat exibido. `onFrame` recebe
   *  RGBA de cima para baixo, com a cor já sem o alpha pré-multiplicado. */
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
  /** Recebe a transformação COLMAP → mundo (centro na origem, "up" em +Y). */
  root: THREE.Group;
  /** A mesma transformação, mas a das poses com que o splat exibido foi
   *  treinado: ao refazer as poses, o splat antigo continua na tela até o
   *  primeiro snapshot do treino novo e não pode herdar o alinhamento novo
   *  (nem a falta dele, que o deixaria de cabeça para baixo). */
  splatRoot: THREE.Group;
  /** `splatRoot` já tem o alinhamento do splat exibido. */
  splatAligned: boolean;
  mesh: SplatMesh | null;
  path: THREE.Object3D | null;
  /** Destaque do ponto sob o cursor (filho de `path`). */
  hover: THREE.Points | null;
  /** Trajetória de retoque (filha de `root`). */
  retouchPath: THREE.Object3D | null;
  /** Pede alguns quadros de renderização (a cena só redesenha quando muda). */
  invalidate: () => void;
}

// Quadros extras após cada mudança: cobrem o amortecimento da órbita e a
// ordenação assíncrona dos splats no Spark.
const SETTLE_FRAMES = 45;

const ACCENT = 0xff9a4d;
/** Raio, em px, para acertar um ponto de câmera com o mouse. */
const PICK_RADIUS = 10;
/** Campo de visão vertical padrão do viewer, em graus. */
const BASE_FOV = 50;
/** Fração máxima da tela que o quadro do enquadramento ocupa. */
const GUIDE_FILL = 0.86;

export function Viewer({ splat, frame, placeholder, onPickCamera, ref }: Props) {
  const { t, locale } = useI18n();
  const tv = t.viewer;
  const host = useRef<HTMLDivElement>(null);
  const ctx = useRef<Ctx | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showCams, setShowCams] = useState(true);
  const [spin, setSpin] = useState(false);
  const [clampView, setClampView] = useState(true);
  const [hasMesh, setHasMesh] = useState(false);
  const [guide, setGuide] = useState<FrameGuide | null>(null);
  const [guideOn, setGuideOn] = useState(false);
  /** Tamanho em px do quadro na tela, com a máscara ligada. */
  const [guideBox, setGuideBox] = useState<{ w: number; h: number } | null>(null);
  const activeGuide = useRef<FrameGuide | null>(null);
  const fitGuideRef = useRef<() => void>(() => {});
  // Os ouvintes do canvas nascem uma vez; leem o estado atual por aqui.
  const frameRef = useRef(frame);
  frameRef.current = frame;
  const pickRef = useRef(onPickCamera);
  pickRef.current = onPickCamera;

  // Cena, renderer e loop: criados uma vez.
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

    // Renderizar só sob demanda libera a GPU para o treino do Brush, que
    // roda em paralelo e disputa o mesmo chip.
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

    controls.addEventListener("change", invalidate);
    renderer.setAnimationLoop(() => {
      if (controls.update() || controls.autoRotate) dirty = SETTLE_FRAMES;
      if (dirty <= 0) return;
      dirty--;
      renderer.render(scene, camera);
    });

    ctx.current = { renderer, scene, camera, controls, root, splatRoot, splatAligned: false, mesh: null, path: null, hover: null, retouchPath: null, invalidate };

    // Ponto de câmera mais próximo do cursor, em pixels da tela.
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
      // Ao trocar de trajetória o destaque novo nasce oculto: refaz mesmo
      // que o índice seja o mesmo.
      if (i === hovered && (i < 0 || c?.hover?.visible)) return;
      hovered = i;
      const f = frameRef.current;
      renderer.domElement.style.cursor = i >= 0 ? "pointer" : "";
      if (!c?.hover) return;
      c.hover.visible = i >= 0 && !!f;
      if (i >= 0 && f) c.hover.position.set(...f.cameras[i]);
      invalidate();
    };
    // Clique = soltar quase no mesmo lugar; arrastar continua girando a órbita.
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => {
      down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
    };
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

    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      ro.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      ctx.current?.mesh?.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      ctx.current = null;
    };
  }, []);

  // Enquadramento a partir das câmeras do COLMAP.
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
      // Liga só frames vizinhos: um salto de índice é um trecho que o COLMAP
      // não conseguiu registrar e deve aparecer como buraco na trajetória.
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
    group.visible = showCams;
    c.root.add(group);
    c.invalidate();
    c.path = group;
    resetView();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  useEffect(() => {
    if (ctx.current?.path) ctx.current.path.visible = showCams;
    ctx.current?.invalidate();
  }, [showCams]);

  useEffect(() => {
    if (ctx.current) ctx.current.controls.autoRotate = spin;
  }, [spin]);

  // Máscara de enquadramento: escurece o que fica fora do quadro do vídeo de
  // retoque. A lente do vídeo pode ser mais aberta que a do viewer; nesse caso
  // o viewer abre o campo de visão até o quadro caber na tela.
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

  // Trava a altura da órbita na faixa que o vídeo filmou (+ folga): fora
  // dela não há dados e o splat mostra buracos e artefatos.
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    const range = clampView && frame ? polarRange(frame) : null;
    c.controls.minPolarAngle = range ? range[0] : 0;
    c.controls.maxPolarAngle = range ? range[1] : Math.PI;
    c.controls.update();
    c.invalidate();
  }, [clampView, frame]);

  // Carrega o .ply (snapshot ou final). Troca sem piscar: o anterior só sai
  // quando o novo terminou de carregar.
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
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
        // Splat novo: nasce com o alinhamento das poses atuais.
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
        // câmera three (olha −Z, +Y em cima) → câmera COLMAP (olha +Z, +Y embaixo)
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
          // Marcas a cada ~8% do clipe: mostram o ritmo e o sentido da câmera.
          const step = Math.max(1, Math.round(pts.length / 12));
          const ticks = new THREE.Points(
            new THREE.BufferGeometry().setFromPoints(pts.filter((_, i) => i % step === 0)),
            pathMaterial(RETOUCH, 0.95, 5),
          );
          const end = new THREE.Points(new THREE.BufferGeometry().setFromPoints([pts[pts.length - 1]]), pathMaterial(RETOUCH, 1, 9));
          const mid = new THREE.Points(new THREE.BufferGeometry().setFromPoints([pts[pts.length >> 1]]), pathMaterial(0xffffff, 1, 10));
          // O início por último, por cima do fim (nas voltas os dois coincidem).
          const start = new THREE.Points(new THREE.BufferGeometry().setFromPoints([pts[0]]), pathMaterial(PATH_START, 1, 11));
          const g = new THREE.Group();
          g.add(line, ticks, end, mid, start);
          // Câmeras nas marcas (e no início e no meio): pirâmide com a base no
          // formato do quadro e um triângulo no lado de cima da imagem.
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
        // A pose pode estar fora da faixa filmada: solta a trava antes de mover
        // (o efeito do estado só roda depois, e o update() já limitaria).
        setClampView(false);
        c.controls.minPolarAngle = 0;
        c.controls.maxPolarAngle = Math.PI;
        c.root.updateMatrixWorld(true);
        c.controls.target.set(0, 0, 0);
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
      setFrameGuide(g) {
        setGuide(g);
      },
      async renderFrames({ poses, size: [w, h], fy, onFrame, signal }) {
        const c = ctx.current;
        if (!c?.mesh) throw new Error("nenhum splat carregado no viewer");
        const f = frameRef.current;
        // Alvo RGBA8 sem codificação sRGB: os valores lidos são os que o
        // shader escreveu (ver o render abaixo).
        const spark = new SparkRenderer({ renderer: c.renderer, target: { width: w, height: h, colorSpace: THREE.NoColorSpace }, minSortIntervalMs: 0 });
        const cam = new THREE.PerspectiveCamera(THREE.MathUtils.radToDeg(2 * Math.atan(h / 2 / fy)), w / h, 0.01, 1000);
        if (f) {
          cam.near = Math.max(f.radius / 500, 0.001);
          cam.far = f.radius * 100;
        }
        cam.updateProjectionMatrix();
        cam.matrixAutoUpdate = false;
        // As linhas de câmera e da trajetória não podem sair no vídeo.
        const hidden = [c.path, c.retouchPath].filter((o): o is THREE.Object3D => !!o && o.visible);
        hidden.forEach((o) => (o.visible = false));
        const row = w * 4;
        const out = new Uint8ClampedArray(w * h * 4);
        try {
          c.root.updateMatrixWorld(true);
          for (let i = 0; i < poses.length; i++) {
            if (signal?.aborted) throw new DOMException("cancelado", "AbortError");
            cam.matrix.copy(c.root.matrixWorld).multiply(c2wFromColmap(poses[i])).multiply(FLIP_YZ);
            cam.matrixWorld.copy(cam.matrix);
            cam.matrixWorldInverse.copy(cam.matrix).invert();
            // A ordenação dos splats é assíncrona: espera a desta pose.
            await spark.update({ scene: c.scene, camera: cam });
            while (spark.sorting || spark.sortDirty) {
              await new Promise((r) => setTimeout(r, 2));
              if (spark.sortDirty && !spark.sorting) await spark.update({ scene: c.scene, camera: cam });
            }
            // Num render target o Spark converte as cores para linear e mistura
            // os splats em espaço linear (encodeLinear, decidido pelo espaço de
            // trabalho do three). Na tela, e no Brush que treinou o splat, a
            // mistura é em sRGB; em linear os escuros desbotam e os claros
            // achatam. Durante o render (síncrono), o espaço de trabalho vira
            // sRGB e o Spark escreve as cores como na tela.
            const working = THREE.ColorManagement.workingColorSpace;
            THREE.ColorManagement.workingColorSpace = THREE.SRGBColorSpace;
            try {
              spark.renderTarget({ scene: c.scene, camera: cam });
            } finally {
              THREE.ColorManagement.workingColorSpace = working;
            }
            const px = await spark.readTarget();
            // WebGL lê de baixo para cima; o vídeo quer de cima para baixo.
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
      // Para os testes automatizados (scripts/web-bench/retouch.mjs).
      if (import.meta.env.DEV) (window as unknown as { __gorbitViewer?: ViewerHandle }).__gorbitViewer = handle;
      return handle;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  /** Alinha o splat a `frame`; sem frame, fica à espera do próximo. */
  function alignSplat(frame: SceneFrame | null) {
    const c = ctx.current!;
    alignTo(c.splatRoot, frame);
    c.splatAligned = !!frame;
    c.invalidate();
  }

  function fitToMesh(mesh: SplatMesh) {
    const c = ctx.current!;
    c.splatRoot.updateMatrixWorld(true);
    const box = mesh.getBoundingBox(true).applyMatrix4(c.splatRoot.matrixWorld);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    c.controls.target.copy(center);
    c.camera.position.copy(center).add(new THREE.Vector3(0, size * 0.25, size * 0.9));
  }

  /** Volta para o ponto de vista do primeiro frame do vídeo, enquadrando o
   *  centro como a filmagem (ou um pouco recuado, sem a lente do vídeo). */
  function resetView() {
    const c = ctx.current;
    if (!c) return;
    c.controls.target.set(0, 0, 0);
    if (frame && frame.cameras.length) {
      c.root.updateMatrixWorld(true);
      const first = new THREE.Vector3(...frame.cameras[0]).applyMatrix4(c.root.matrixWorld);
      // A lente daqui é fixa (50°); a do vídeo pode ser bem mais fechada
      // (teleobjetiva). Aproxima até o quadro do vídeo caber justo na tela.
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
/** Início da trajetória de retoque. */
const PATH_START = 0x6fe3a1;

/** Material da trajetória de retoque: desenhada por cima do splat (que não
 *  escreve profundidade), mas apagada no trecho que passa atrás do centro da
 *  órbita (a origem do mundo). Sem isso uma volta em torno do sujeito, vista
 *  quase de lado, parece uma diagonal cruzando o corpo. `size` > 0: pontos. */
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
/** Troca entre os eixos de câmera do three (+Y em cima, olha −Z) e do COLMAP. */
const FLIP_YZ = new THREE.Matrix4().makeScale(1, -1, -1);

/** Câmera → mundo (COLMAP) de uma pose mundo → câmera. */
function c2wFromColmap({ qvec: [w, x, y, z], tvec }: ColmapPose) {
  const r = new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(x, y, z, w));
  const t = new THREE.Matrix4().makeTranslation(tvec[0], tvec[1], tvec[2]);
  return t.multiply(r).invert();
}

const centerOf = (p: ColmapPose) => new THREE.Vector3().setFromMatrixPosition(c2wFromColmap(p));

/** Segmentos (pares de pontos) de uma câmera desenhada: pirâmide da posição
 *  até o quadro a `depth` à frente, o quadro e um triângulo no lado de cima da
 *  imagem. Coordenadas do COLMAP (+Z à frente, +Y para baixo). */
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

/** Transformação COLMAP → mundo: centro na origem, "up" em +Y. */
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
