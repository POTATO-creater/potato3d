// Sessão WebXR do visualizador de links (VR nos óculos, AR no celular):
// disponibilidade, entrar/sair, posicionar a cena e andar com os controles.
// Porte do player do WebGaussian (xr-session.ts).
//
// A cena não é mexida; dois nós criados junto com o app fazem o trabalho:
// - `rig`: pai da câmera. A pose dos óculos se compõe com ele, então andar
//   (analógico) e virar (snap turn) movem o rig.
// - `stage`: pai do splat. Ao entrar, ele escala a cena para uma maquete de
//   ~1,4 m em pé na frente do usuário, virada para o lado que o vídeo filmou;
//   ao sair, volta à identidade para a câmera orbital.
//
// Diferente do WebGaussian (um objeto isolado), aqui a cena tem fundo e
// splats soltos em volta das câmeras. No óculos isso pesa: splats grandes
// cobrindo a visão inteira, duas vezes (um olho cada), são o que mais custa
// num Quest, e de perto viram manchas de cor. Por isso, só durante a sessão,
// o que fica fora de uma esfera em volta do sujeito é descartado no shader
// (`crop`), além da ordenação radial e da foveação (ver `XrOptions`).

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

/** Ajustes de desempenho do XR (padrões em main.ts, sobrepostos pela URL). */
export interface XrOptions {
  /** Resolução do framebuffer do XR (1 = a recomendada pelo aparelho). */
  framebufferScale: number;
  /** Foveação fixa, 0–1: menos resolução nas bordas da lente. */
  foveation: number;
  /** Ordena os splats pela distância à cabeça, não pela direção do olhar:
   *  menos artefatos ao virar a cabeça (recomendação do PlayCanvas). */
  radialSort: boolean;
}

/** O que vira maquete, no espaço do `stage`. */
export interface XrContent {
  center: Vec3;
  /** Raio que vira `STAGE_RADIUS`. */
  radius: number;
  /** Giro (graus, em torno de +Y) que deixa o lado filmado de frente. */
  yaw: number;
  /** Raio além do qual os splats somem durante a sessão; 0 não recorta. */
  crop: number;
}

/** Raio da maquete no XR (metros). */
const STAGE_RADIUS = 0.7;
/** Altura do centro da cena acima do chão (espaços `local-floor`). */
const STAGE_CENTER_HEIGHT = 1.3;
/** Folga entre o usuário e a borda mais próxima da maquete (metros). */
const STAGE_CLEARANCE = 0.8;
/** Velocidade do analógico e da subida/descida (m/s, já em escala humana). */
const MOVE_SPEED = 1.5;
const LIFT_SPEED = 1.5;
const MOVE_DEADZONE = 0.15;
/** Snap turn: dispara no limiar de engate, rearma abaixo do de soltura. */
const SNAP_TURN_DEGREES = 45;
const SNAP_ENGAGE = 0.6;
const SNAP_RELEASE = 0.3;

export interface XrCallbacks {
  /** Mudou a disponibilidade de immersive-vr / immersive-ar (também no início). */
  onAvailability(vr: boolean, ar: boolean): void;
  /** Sessão começou (true) ou terminou (false). */
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
      xr.on("error", (e: Error) => console.warn("[share] erro no XR", e)),
    ];
    this.detach.push(() => handles.forEach((h) => h.off()));
    emit();
  }

  get active() {
    return this.xr?.active ?? false;
  }

  /** Entra no XR; tem que ser chamado num gesto do usuário (clique). Prefere
   *  `local-floor` (y = 0 no chão de verdade) e cai para `local`. */
  start(mode: XrMode) {
    if (!this.xr?.supported || this.xr.active) return;
    const type = mode === "ar" ? XRTYPE_AR : XRTYPE_VR;
    this.startWith(type, XRSPACE_LOCALFLOOR, () => this.startWith(type, XRSPACE_LOCAL));
  }

  end() {
    if (this.xr?.active) this.xr.end();
  }

  /** O que vira maquete (ver `XrContent`). */
  setContent(content: XrContent) {
    this.content = { ...content, center: content.center.clone() };
    if (this.active) this.placeStage();
  }

  /** A entidade do splat, para o recorte durante a sessão. */
  setSplat(entity: Entity) {
    this.splat = entity;
    if (this.active) this.placeStage();
  }

  /** Monta a maquete fora do XR, para conferir no navegador o que o óculos
   *  vai mostrar (`?xrPreview=1`). Devolve onde fica a cabeça e o centro. */
  preview() {
    this.app.scene.gsplat.radialSorting = this.options.radialSort;
    this.placeStage(STAGE_CENTER_HEIGHT);
    return { head: new Vec3(0, 1.6, 0), target: new Vec3(0, STAGE_CENTER_HEIGHT, -(STAGE_RADIUS + STAGE_CLEARANCE)) };
  }

  /** Andar e virar com os controles; chamar a cada quadro durante a sessão. */
  update(dt: number) {
    if (!this.xr?.active || dt <= 0) return;
    let moveX = 0;
    let moveZ = 0;
    let turn = 0;
    let lift = 0;
    for (const source of this.xr.input.inputSources) {
      const axes = source.gamepad?.axes;
      if (!axes?.length) continue;
      // O mapeamento padrão põe o analógico em [2,3]; quem só tem touchpad usa [0,1].
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
        else console.warn("[share] não foi possível entrar no XR", err);
      },
    });
  }

  private onStart() {
    // O AR compõe sobre o mundo real: fundo transparente durante a sessão.
    const camera = this.cameraEntity.camera;
    if (camera && this.xr?.type === XRTYPE_AR) {
      this.savedClearColor.copy(camera.clearColor);
      camera.clearColor = new Color(0, 0, 0, 0);
    }
    this.snapArmed = true;
    this.app.scene.gsplat.radialSorting = this.options.radialSort;
    // null: o aparelho não tem foveação fixa.
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

  /** Escala a cena para a maquete e a põe de pé na frente da pose inicial:
   *  raio → `STAGE_RADIUS`, centro na altura do peito (com chão) e
   *  `STAGE_CLEARANCE` além do alcance do braço, em -Z, com o lado filmado
   *  virado para o usuário. */
  private placeStage(height = this.xr?.spaceType === XRSPACE_LOCALFLOOR ? STAGE_CENTER_HEIGHT : 0) {
    const c = this.content;
    if (!c) return;
    const scale = STAGE_RADIUS / (c.radius || 1);
    const target = new Vec3(0, height, -(STAGE_RADIUS + STAGE_CLEARANCE));
    const rotation = new Quat().setFromEulerAngles(0, c.yaw, 0);
    // O centro do conteúdo tem que cair em `target`: pos = target − R·(c·s).
    const offset = rotation.transformVector(c.center.clone().mulScalar(scale));
    this.stage.setLocalScale(scale, scale, scale);
    this.stage.setLocalEulerAngles(0, c.yaw, 0);
    this.stage.setLocalPosition(target.clone().sub(offset));
    // O recorte roda no buffer de trabalho do PlayCanvas, já em coordenadas
    // de mundo: o centro do sujeito é `target` e o raio vem escalado.
    if (this.splat?.gsplat && c.crop > 0) this.splat.gsplat.setWorkBufferModifier(cropShader(target, c.crop * scale));
  }

  /** Giro em degraus em volta da cabeça: vira sem enjoo. */
  private snapTurn(turn: number) {
    const m = Math.abs(turn);
    if (m < SNAP_RELEASE) {
      this.snapArmed = true;
      return;
    }
    if (!this.snapArmed || m < SNAP_ENGAGE) return;
    this.snapArmed = false;
    const degrees = -Math.sign(turn) * SNAP_TURN_DEGREES;
    // Gira o rig em torno da posição da cabeça, para ela ficar no lugar.
    const head = this.cameraEntity.getPosition().clone();
    const offset = this.rig.getPosition().clone().sub(head);
    new Quat().setFromEulerAngles(0, degrees, 0).transformVector(offset, offset);
    this.rig.setPosition(offset.add(head));
    this.rig.rotate(0, degrees, 0);
  }

  /** Desliza no plano relativo à cabeça; o Y do analógico direito sobe/desce. */
  private glide(moveX: number, moveZ: number, lift: number, dt: number) {
    const moving = Math.hypot(moveX, moveZ) >= MOVE_DEADZONE;
    if (!moving && Math.abs(lift) < MOVE_DEADZONE) return;
    const forward = flat(this.cameraEntity.forward);
    const right = flat(this.cameraEntity.right);
    const step = MOVE_SPEED * dt;
    const vertical = Math.abs(lift) >= MOVE_DEADZONE ? -lift * LIFT_SPEED * dt : 0;
    const horizontal = moving && forward && right;
    // Analógico para cima (Y negativo) vai para a frente.
    this.rig.translate(
      horizontal ? (right.x * moveX - forward.x * moveZ) * step : 0,
      vertical,
      horizontal ? (right.z * moveX - forward.z * moveZ) * step : 0,
    );
  }
}

/** Descarta (escala e opacidade zero) os splats fora da esfera. */
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

/** Projeta a direção no plano XZ e normaliza (null se for vertical). */
function flat(dir: Vec3): Vec3 | null {
  const v = new Vec3(dir.x, 0, dir.z);
  const l = v.length();
  return l > 1e-4 ? v.mulScalar(1 / l) : null;
}
