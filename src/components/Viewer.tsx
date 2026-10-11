import { useEffect, useRef, useState } from "react";
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
}

interface Ctx {
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  /** Recebe a transformação COLMAP → mundo (centro na origem, "up" em +Y). */
  root: THREE.Group;
  mesh: SplatMesh | null;
  path: THREE.Object3D | null;
  /** Pede alguns quadros de renderização (a cena só redesenha quando muda). */
  invalidate: () => void;
}

// Quadros extras após cada mudança: cobrem o amortecimento da órbita e a
// ordenação assíncrona dos splats no Spark.
const SETTLE_FRAMES = 45;

const ACCENT = 0xff9a4d;

export function Viewer({ splat, frame, placeholder }: Props) {
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
    scene.add(root);

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

    ctx.current = { renderer, camera, controls, root, mesh: null, path: null, invalidate };
    return () => {
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
    }
    if (!frame) {
      c.root.position.set(0, 0, 0);
      c.root.quaternion.identity();
      return;
    }
    const up = new THREE.Vector3(...frame.up).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(up, new THREE.Vector3(0, 1, 0));
    const center = new THREE.Vector3(...frame.center).applyQuaternion(q);
    c.root.quaternion.copy(q);
    c.root.position.copy(center.negate());

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
      group.add(line, dots);
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
        c.root.remove(c.mesh);
        c.mesh.dispose();
        c.mesh = null;
        c.invalidate();
        setHasMesh(false);
      }
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
        c.root.add(mesh);
        c.invalidate();
        c.mesh = mesh;
        if (prev) {
          c.root.remove(prev);
          prev.dispose();
        }
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

  function fitToMesh(mesh: SplatMesh) {
    const c = ctx.current!;
    const box = mesh.getBoundingBox(true).applyMatrix4(c.root.matrixWorld);
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
        <button className={`tool ${showCams ? "on" : ""}`} onClick={() => setShowCams((v) => !v)} disabled={!frame}>
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
