import * as THREE from "three";

/** How an exported mesh (.glb) is drawn in the viewer, to judge its quality:
 *  its own colours, plain clay (shape only), clay with the triangle edges,
 *  the edges alone (hidden lines removed) or the normals. */
export type MeshView = "original" | "clay" | "clayWire" | "wire" | "normals";
export const MESH_VIEWS: MeshView[] = ["original", "clay", "clayWire", "wire", "normals"];

/** Back faces in clay: a hole or a flipped triangle shows up in this colour. */
const BACK_FACE = "vec3(0.85, 0.32, 0.27)";

let mats: Record<"clay" | "depth" | "edgesOnClay" | "edges" | "normals", THREE.Material> | null = null;

/** Shared by every mesh and never disposed (`userData.shared`). */
function materials() {
  if (mats) return mats;
  // pushed back so the edges drawn over it win the depth test
  const behind = { polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 };
  const clay = new THREE.MeshMatcapMaterial({ matcap: clayMatcap(), side: THREE.DoubleSide, ...behind });
  clay.onBeforeCompile = (s) => {
    s.fragmentShader = s.fragmentShader.replace(
      "#include <color_fragment>",
      `#include <color_fragment>\n if (!gl_FrontFacing) diffuseColor.rgb = ${BACK_FACE};`,
    );
  };
  // depth only: hides the edges behind the surface
  const depth = new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide, ...behind });
  const edgesOnClay = new THREE.MeshBasicMaterial({ color: 0x15171a, wireframe: true, transparent: true, opacity: 0.45, depthWrite: false });
  const edges = new THREE.MeshBasicMaterial({ color: 0xa9c4dc, wireframe: true, transparent: true, opacity: 0.7, depthWrite: false });
  const normals = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
  mats = { clay, depth, edgesOnClay, edges, normals };
  for (const m of Object.values(mats)) m.userData.shared = true;
  return mats;
}

/** Clay look that follows the camera (a matcap): soft key light from the
 *  upper left, a fill from below and a faint rim, on a warm grey. */
function clayMatcap() {
  const n = 256;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = n;
  const g = canvas.getContext("2d")!;
  const img = g.createImageData(n, n);
  const key = new THREE.Vector3(-0.45, 0.55, 0.7).normalize();
  const fill = new THREE.Vector3(0.3, -0.6, 0.5).normalize();
  const half = new THREE.Vector3(0, 0, 1).add(key).normalize();
  const nrm = new THREE.Vector3();
  const base = [0.82, 0.78, 0.72];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const nx = ((x + 0.5) / n) * 2 - 1;
      const ny = 1 - ((y + 0.5) / n) * 2;
      const r2 = nx * nx + ny * ny;
      nrm.set(nx, ny, Math.sqrt(Math.max(0, 1 - Math.min(r2, 1))));
      const d = Math.max(0, nrm.dot(key));
      const shade = 0.3 + 0.7 * d + 0.12 * Math.max(0, nrm.dot(fill)) + 0.18 * Math.pow(1 - nrm.z, 3);
      const spec = 0.12 * Math.pow(Math.max(0, nrm.dot(half)), 24);
      const i = (y * n + x) * 4;
      for (let c = 0; c < 3; c++) img.data[i + c] = Math.min(255, Math.round((base[c] * shade + spec) * 255));
      img.data[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** Draws every mesh under `root` in `view`. The original materials are kept
 *  in `userData`; the edges are a child mesh sharing the geometry. */
export function paintMesh(root: THREE.Object3D, view: MeshView) {
  const m = materials();
  const meshes: THREE.Mesh[] = [];
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh && !o.userData.edges) meshes.push(o as THREE.Mesh);
  });
  for (const mesh of meshes) {
    mesh.userData.material ??= mesh.material;
    if (view !== "original" && !mesh.geometry.getAttribute("normal")) mesh.geometry.computeVertexNormals();
    mesh.material =
      view === "clay" || view === "clayWire" ? m.clay : view === "wire" ? m.depth : view === "normals" ? m.normals : mesh.userData.material;
    let edges = mesh.children.find((c) => c.userData.edges) as THREE.Mesh | undefined;
    const showEdges = view === "clayWire" || view === "wire";
    if (showEdges && !edges) {
      edges = new THREE.Mesh(mesh.geometry, m.edges);
      edges.userData.edges = true;
      edges.renderOrder = 1;
      mesh.add(edges);
    }
    if (edges) {
      edges.visible = showEdges;
      edges.material = view === "wire" ? m.edges : m.edgesOnClay;
    }
  }
}

/** Triangles and vertices of the meshes under `root`. */
export function meshStats(root: THREE.Object3D) {
  let triangles = 0;
  let vertices = 0;
  root.traverse((o) => {
    const g = (o as THREE.Mesh).isMesh && !o.userData.edges ? (o as THREE.Mesh).geometry : null;
    if (!g) return;
    const pos = g.getAttribute("position");
    vertices += pos.count;
    triangles += (g.index ? g.index.count : pos.count) / 3;
  });
  return { triangles, vertices };
}
