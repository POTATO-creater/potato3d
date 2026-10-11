// Catálogo de referência: os outros idiomas têm que ter exatamente esta forma.

import { fmtCount } from "../lib/format";

/** Parâmetros de uma `Msg`, com as mensagens aninhadas já traduzidas. */
export type Args = Record<string, string | number>;

const FEATURES: Record<string, string> = { sift: "SIFT", siftPlus: "SIFT reforçado", aliked: "ALIKED + LightGlue" };
const features = (id: string | number) => FEATURES[id] ?? String(id);

export const ptBR = {
  language: "Português (Brasil)",

  /** Nomes curtos das etapas (botões e listas). */
  stages: { frames: "Frames", masks: "Recorte", poses: "Poses", train: "Treino" },
  /** Títulos da faixa de etapas. */
  stageTitles: { frames: "Frames", masks: "Recorte", poses: "Poses de câmera", train: "Treino do splat" },

  common: {
    cancel: "Cancelar",
    language: "Idioma",
  },

  app: {
    resizeSidebar: "Arraste para redimensionar · duplo clique volta ao tamanho padrão",
    tabs: { viewer: "3D", frames: "Frames", log: "Log" },
    readyIn: (d: string) => `Pronto em ${d}`,
    videoError: (e: string) => `Não foi possível ler o vídeo: ${e}`,
    imagesError: (e: string) => `Não foi possível usar as imagens: ${e}`,
    failed: "Falhou.",
    failedAt: (stage: string) => `Falhou na etapa ${stage}.`,
    logError: (e: string) => `ERRO: ${e}`,
    exported: (name: string) => `Exportado · ${name}`,
    hero: "Vídeo orbital → Gaussian Splatting",
    optional: "opcional",
    dropToStart: "Arraste um vídeo, imagens ou uma pasta de frames para a janela para começar.",
    maskModel: "Modelo de recorte",
    masker: "Recortador",
  },

  platform: {
    browserDecoder: "decodificador do navegador",
    yourImages: "suas imagens · nitidez",
    solidColor: "cor sólida",
    webPoses: "AKAZE · WebGPU · SfM incremental",
    hwDecode: "decodificação por hardware",
    globalMapper: "mapeador global",
    webgpuChrome: "Chrome ou Edge com WebGPU",
    webgpuBrowser: "navegador com WebGPU",
    installTools: "Instalar ferramentas",
    packageManager: (tool: string) => `instale ${tool} pelo gerenciador de pacotes`,
    fileManager: "gerenciador de arquivos",
    download: (what?: string) => `Baixar${what ? ` ${what}` : ""}`,
    reveal: (fileManager: string, what?: string) => `Mostrar${what ? ` ${what}` : ""} no ${fileManager}`,
    blurbs: {
      web: {
        frames: "o navegador decodifica o vídeo (ou lê as suas imagens) e escolhe os frames mais nítidos.",
        masks: "isola o sujeito para o splat sair sem o fundo.",
        poses: "features AKAZE, matching circular na GPU (fecha a órbita) e SfM incremental.",
        train: "Brush em WebAssembly + WebGPU, com prévias ao vivo aqui.",
      },
      desktop: {
        frames: "ffmpeg decodifica o vídeo por hardware (ou lê as suas imagens) e escolhe os frames mais nítidos.",
        masks: "isola o sujeito para o splat sair sem o fundo.",
        poses: "COLMAP com matching circular (fecha a órbita) e mapeador global.",
        train: "Brush na GPU, com prévias ao vivo aqui.",
      },
    },
  },

  presets: {
    preview: { label: "Prévia", hint: "Confere rápido se o vídeo reconstrói" },
    balanced: { label: "Equilibrado", hint: "Bom ponto de partida para a maioria dos vídeos" },
    quality: { label: "Qualidade", hint: "Mais frames, 30k passos e perda SSIM; bem mais lento no Mac" },
  },

  sidebar: {
    video: "Vídeo",
    images: "Imagens",
    source: "Vídeo ou imagens",
    chooseFolder: "Pasta…",
    chooseFolderTitle: "Usar uma pasta com os frames já extraídos",
    imageCount: (n: number) => `${n} imagens`,
    imagesHelp:
      "Entram na ordem dos nomes, que deve ser a da órbita. Com mais imagens que o número de frames, fica a mais nítida de cada trecho.",
    changeVideo: "Trocar…",
    chooseVideo: "Escolher…",
    trim: "Trecho",
    startHere: "Início aqui",
    endHere: "Fim aqui",
    wholeVideo: "Vídeo todo",
    trimHelp:
      'Arraste as alças laranja na linha do tempo ou use os botões. Use só o trecho em que o objeto continua consistente: vídeo de IA costuma "derreter" no fim da órbita.',
    dropTitle: "Arraste um vídeo orbital, imagens ou uma pasta de frames",
    dropHint: "ou clique para escolher (.mp4, .mov, .webm, .jpg, .png)",
    project: "Projeto",
    importProject: "Importar…",
    importTitle: "Importa uma pasta de projeto do Gausify Orbit ou um dataset COLMAP (images/ + sparse/0)",
    changeProject: "Alterar…",
    pickVideoFirst: "Escolha um vídeo ou imagens primeiro",
    quality: "Qualidade",
    custom: "personalizado",
    advanced: "Parâmetros avançados",
    frames: "Frames",
    maxSide: "Lado máx.",
    candidates: "Candidatos/frame",
    candidatesHint: "escolhe o mais nítido",
    poses: "Poses",
    posesColmap: "Poses (COLMAP)",
    features: "Features",
    featuresHint: {
      sift: "se registrar < 95% das câmeras, repete com o reforçado",
      siftPlus: "acha pontos em cabelo, pele e fundo desfocado (~2× o tempo)",
      aliked: "registra bem, mas ~10× mais lento na CPU",
    },
    featureOptions: {
      sift: { label: "SIFT (rápido)" },
      siftPlus: { label: "SIFT reforçado", hint: "~2× o tempo" },
      aliked: { label: "ALIKED + LightGlue", hint: "~10× mais lento na CPU" },
    },
    matchWindow: "Vizinhos casados",
    camera: "Câmera",
    simplePinhole: "Pinhole simples",
    pinhole: "Pinhole (fx ≠ fy)",
    closedLoop: "Órbita fechada (360°)",
    closedLoopHint: "casa o fim com o começo",
    flatOrbit: "Forçar órbita plana",
    flatOrbitHint: "câmeras na mesma altura; só para vídeos que giram em nível",
    training: "Treino (Brush)",
    steps: "Passos",
    shDegree: "Grau SH",
    maxSplats: "Máx. splats",
    maxResolution: "Resolução máx.",
    previewEvery: "Prévia a cada",
    stepsSuffix: "passos",
    ssim: "Perda SSIM",
    ssimHint: "detalhes um pouco mais finos, mas ~3× mais lento no Mac",
    smooth: "Suavizar áreas não filmadas",
    smoothHint: "menos riscos e cores estranhas no topo; −1,3 dB de detalhe",
    psnr: "Medir qualidade (PSNR)",
    psnrHint: "separa 1 a cada 16 frames para avaliação",
    runFrom: "Executar a partir de",
    needsFrames: "Requer frames extraídos",
    enableMasks: "Ative o recorte do sujeito",
    needsPoses: "Requer poses (e recortes, se ativados)",
    run: "Gerar splat",
  },

  masks: {
    title: "Recorte do sujeito",
    enable: "Isolar o sujeito (remover o fundo)",
    enableHint: "o splat sai só com a pessoa ou objeto, sem paredes e riscos em volta",
    methods: {
      chroma: {
        label: "Cor sólida",
        hint:
          "Chroma key: instantâneo. Para fundo de cor única (tela verde, azul…). Se o sujeito tiver partes " +
          "da cor do fundo, prefira o RMBG com o despill ligado.",
      },
      rmbg: {
        label: "RMBG",
        hint:
          "RMBG-1.4 na GPU: bordas mais limpas, qualquer sujeito. Baixa 176 MB na primeira vez. " +
          "Licença da BRIA AI só para uso não comercial.",
      },
      sam2: {
        label: "SAM 2.1",
        hint:
          "SAM 2.1 da Meta na GPU: segue o mesmo objeto por todos os frames, sem variar entre um e outro. " +
          "Marque o sujeito no vídeo; sem marcação, usa o que está no centro. Baixa 84 MB na primeira vez (165 MB em GPUs sem fp16). Licença Apache-2.0.",
      },
      people: { label: "Pessoas", hint: "MODNet na GPU: rápido, mas só reconhece pessoas." },
      general: { label: "Objetos", hint: "BiRefNet: qualquer sujeito, bem mais lento." },
      auto: { label: "Vision", hint: "Recorte nativo do macOS, na GPU: qualquer sujeito." },
    },
    bgColor: "Cor do fundo",
    autoColor: "Detectar a cor pelas bordas",
    autoColorHint: "mediana das bordas de cada frame",
    tolerance: "Tolerância",
    softness: "Suavização da borda",
    contiguous: "Só regiões ligadas à borda",
    contiguousHint: "não fura partes do sujeito com a mesma cor do fundo",
    despill: "Remover reflexo do fundo verde/azul (despill)",
    despillHint: "tira do sujeito inteiro a luz rebatida pela tela; desligue se ele tiver a cor do fundo",
    despillStrength: "Intensidade do despill",
    forPoses: "Usar o recorte também nas poses",
    forPosesHint: "ajuda se o fundo muda entre frames; atrapalha em sujeitos com pouca textura",
    prompt: {
      none: "Sem marcação: segue o objeto no centro do primeiro frame.",
      marked: (time: string, what: string) => `Sujeito marcado em ${time} (${what}).`,
      box: "caixa",
      points: (n: number) => (n === 1 ? "1 ponto" : `${n} pontos`),
      clear: "Limpar",
      howTo: "No vídeo (ou nas imagens), toque no alvo ⌖ e arraste uma caixa em volta do sujeito ou clique nele. Alt+clique marca o fundo; Shift+clique apaga um ponto ou a caixa.",
      outside: "A marcação ficou fora do trecho e será ignorada (vale o objeto do centro). Marque de novo dentro dele.",
    },
    preview: {
      title: "Prévia",
      at: (time: string) => `quadro em ${time}`,
      show: "Mostrar",
      hide: "Ocultar",
      load: "Gerar prévia (baixa o modelo na primeira vez)",
      downloading: (done: number, total: number) => `Baixando o modelo: ${done}/${total} MB`,
      working: "Calculando…",
      paused: "Prévia pausada durante o processamento.",
      failed: (e: string) => `Prévia indisponível: ${e}`,
      legend: "O xadrez é o fundo que sai. Mude a marcação e a prévia acompanha.",
    },
  },

  export: {
    title: "Exportação",
    format: "Formato",
    formatHintSog: "texturas WebP quantizadas, ~5–15× menor; abre no Spark, PlayCanvas e SuperSplat",
    formatHintPly: "float32 sem perdas; compatível com qualquer visualizador",
    sog: "SOG comprimido",
    ply: "PLY",
    reduce: "Reduzir",
    reduceHint: "funde splats parecidos",
    minOpacity: "Opacidade mín.",
    minOpacityHint: "0 desliga",
    shBands: "Bandas SH",
    shBandsHint: "brilho que muda com o ângulo; cortar reduz bastante o arquivo",
    /** Rótulo de cada grau SH (minúsculo; o menu capitaliza). */
    shLabels: ["só cor (grau 0)", "grau 1", "grau 2", "grau 3"],
    /** O que cada grau mantém, para o menu. */
    shHints: ["sem reflexos, o menor arquivo", "reflexos suaves", "reflexos mais nítidos"],
    keep: (label: string) => `Manter (${label})`,
    run: "Exportar",
    runApprox: (count: string) => `Exportar ~${count} splats`,
    smaller: (ratio: string) => `${ratio}× menor`,
    pruned: (byOpacity: string, byMerge: string) => `Poda por opacidade tirou ${byOpacity}; a fusão, ${byMerge}.`,
    viewOriginal: "Ver original",
    view3d: "Ver no 3D",
  },

  share: {
    button: "Compartilhar",
    intro: (size: string) =>
      `Envia este arquivo (${size}) para a nuvem e gera um link privado que abre o splat no navegador, no celular ou em óculos de VR. Só quem tiver o link vê, e ele expira sozinho.`,
    create: "Gerar link",
    uploading: (pct: number) => `Enviando… ${pct}%`,
    copy: "Copiar",
    copied: "Copiado",
    open: "Abrir",
    expires: (when: string) => `Expira às ${when}`,
    revoke: "Revogar",
    revokeTitle: "Apaga o arquivo da nuvem agora; o link para de funcionar",
    never: "Não expira. Revogue quando não precisar mais.",
    ttlLabel: "Validade",
    ttl: (v: number | "never") =>
      v === "never"
        ? "Sem expiração"
        : v < 24
          ? `${v} horas`
          : v % 8760 === 0
            ? v === 8760 ? "1 ano" : `${v / 8760} anos`
            : v === 24 ? "1 dia" : `${v / 24} dias`,
    haveKey: "Tenho uma chave de acesso",
    keyPlaceholder: "Chave de acesso",
    keySave: "Salvar",
    forgetKey: "Esquecer chave",
    forgetKeyTitle: "Tira a chave deste aparelho; os links voltam à validade padrão",
    qrHint: "Aponte a câmera do celular para abrir. Nos óculos de VR, abra o link no navegador deles e toque em VR.",
    errors: {
      too_large: (max: string) => `O arquivo passa do limite de ${max}. Exporte em SOG ou reduza os splats.`,
      bad_format: "O servidor recusou o arquivo: só .sog e .ply.",
      rate_limited: "Muitos envios seguidos; tente de novo em um minuto.",
      network: "Sem conexão com o servidor de compartilhamento.",
      forbidden: "Chave de acesso inválida; digite de novo.",
      bad_ttl: "O servidor recusou essa validade.",
      server: "O servidor não conseguiu criar o link.",
    },
  },

  /** Página do link compartilhado (web/share.html). */
  shareViewer: {
    title: "Splat compartilhado",
    opening: "Abrindo…",
    loading: (pct: number, size: string) => `Baixando ${pct}% de ${size}`,
    preparing: "Preparando os splats",
    expiresIn: (left: string) => `expira em ${left}`,
    expired: "link expirado",
    goneTitle: "Este link expirou ou não existe",
    goneBody: "O link pode ter vencido ou sido revogado. Peça um novo a quem enviou.",
    makeYours: "Criar o seu no Gausify Orbit",
    loadError: (e: string) => `Não foi possível abrir o splat: ${e}`,
    spin: "Girar",
    recenter: "Recentrar",
    fullscreen: "Tela cheia",
    vr: "VR",
    vrTitle: "Ver em realidade virtual",
    ar: "AR",
    arTitle: "Ver em realidade aumentada",
    hintMouse: "Arraste para girar · roda para zoom · botão direito para mover · WASD para andar",
    hintTouch: "Arraste para girar · pinça para zoom · dois dedos para mover",
  },

  frames: {
    empty: "Os frames selecionados aparecem aqui depois da primeira etapa.",
    count: (n: number) => `${n} frames`,
    lowSharpness: (n: number) => `${n} com menos da metade da nitidez mediana`,
    uniform: "Nitidez uniforme",
    showingCut: "Mostrando recorte",
    showCut: "Mostrar recorte",
    sharpness: "nitidez",
    prev: "Anterior",
    next: "Próximo",
    close: "Fechar",
    noLog: "Nenhum log ainda.",
  },

  stageStrip: {
    selected: "Selecionados",
    medianSharpness: "Nitidez mediana",
    cuts: "Recortes",
    coverage: "Sujeito ocupa",
    noSubject: "Sem sujeito",
    background: "Fundo",
    cameras: "Câmeras",
    points: "Pontos",
    reprojError: "Erro reproj.",
    splats: "Splats",
    speed: "Velocidade",
    stepsPerSec: (n: string) => `${n} passos/s`,
    remaining: "Restante",
    psnrTitle: "Medido em frames que não entram no treino. >30 dB é ótimo; <25 dB indica vídeo inconsistente.",
  },

  setup: {
    title: "Primeiro uso: faltam ferramentas",
    recheck: "Verificar de novo",
    nvidia: "GPU NVIDIA detectada: o COLMAP com CUDA usa a placa para extrair e casar features. ",
    noNvidia: "Nenhuma GPU NVIDIA detectada: o COLMAP roda na CPU. ",
    rmbgLicense: "O RMBG-1.4 é da BRIA AI, com licença só para uso não comercial. ",
    storedInBrowser: "Os arquivos são conferidos por SHA-256 e ficam guardados no navegador.",
    storedLocally: "Os arquivos são conferidos por SHA-256 e ficam na pasta de dados do usuário.",
    install: (size: string) => `Instalar ferramentas (${size})`,
    checking: "Verificando o que precisa ser baixado…",
    planError: (e: string) => `Não foi possível preparar a instalação: ${e}`,
    queued: "na fila",
    extracting: "extraindo…",
    ready: "pronto",
    /** Seguido de `pnpm setup:tools` em código. */
    brushDev: "O Brush deveria vir junto com o app. Em desenvolvimento, rode",
    visionNote: "O recortador (Vision) vem com o app e exige macOS 14 ou superior.",
    /** Rótulos dos modelos baixáveis (os demais pacotes usam o nome da ferramenta). */
    packages: {
      rmbg: "Recorte RMBG-1.4",
      modnet: "Recorte de pessoas (MODNet)",
      birefnet: "Recorte de objetos (BiRefNet)",
    } as Record<string, string>,
  },

  player: {
    dragStart: "Arraste para definir o início",
    dragEnd: "Arraste para definir o fim",
    pause: "Pausar (espaço)",
    play: "Reproduzir (espaço)",
    prevFrame: "Frame anterior (←)",
    nextFrame: "Próximo frame (→)",
    prevImage: "Imagem anterior (←)",
    nextImage: "Próxima imagem (→)",
    loop: "Repetir só o trecho selecionado",
    mark: "Marcar o sujeito para o SAM 2.1",
    markDone: "Concluir a marcação",
    markHint: "Arraste uma caixa ou clique no sujeito · Alt+clique: fundo · Shift+clique: apaga",
    markOutside: "Fora do trecho: volte para dentro dele para marcar",
  },

  viewer: {
    preview: (step: string) => `Prévia · passo ${step}`,
    final: "Resultado final",
    loading: "Carregando splat…",
    loadError: (e: string) => `Erro ao carregar: ${e}`,
    cameras: "Câmeras",
    camerasTitle: "Clique num ponto para ver o frame daquela câmera",
    lockAngle: "Travar ângulo",
    lockAngleTitle: "Limita a altura da câmera às regiões que o vídeo filmou",
    spin: "Girar",
    recenter: "Recentrar",
  },

  /** Versão web: aviso de versão nova baixada pelo service worker. */
  update: {
    available: "Há uma nova versão do Gausify Orbit pronta para usar.",
    apply: "Atualizar agora",
    later: "Depois",
    busy: "Disponível quando o processamento terminar",
  },

  /** Mensagens de progresso (backend e reducer), por chave da `Msg`. */
  msg: {
    "stage.saved": () => "Resultado salvo no projeto",
    "stage.queued": () => "Na fila",
    "stage.starting": () => "Iniciando",
    "stage.failed": () => "Falhou",
    "stage.cancelled": () => "Cancelado",

    "frames.reading": () => "Lendo vídeo",
    "frames.decoding": () => "Decodificando vídeo",
    "frames.decodingSharpness": () => "Decodificando e medindo nitidez",
    "frames.sharpness": () => "Medindo nitidez",
    "frames.converting": () => "Convertendo imagens",
    "frames.selected": (a: Args) => `${a.n} frames selecionados`,

    "masks.detecting": () => "Detectando o sujeito",
    "masks.downloading": (a: Args) => `Baixando ${a.model}: ${a.done}/${a.total} MB`,
    "masks.cutting": (a: Args) => `Recortando ${a.i}/${a.n}`,
    "masks.done": (a: Args) => `${a.n} recortes · ${a.backend}`,
    "masks.doneChroma": (a: Args) => `${a.n} recortes · cor sólida`,
    "masks.disabled": () => "Recorte desativado",
    "masks.alphaFrames": (a: Args) => `Frames com fundo transparente ${a.i}/${a.n}`,

    "poses.reused": () => "Poses anteriores reaproveitadas",
    "poses.extracting": (a: Args) => `Extraindo features (${features(a.features)})`,
    "poses.extractingN": (a: Args) => `Extraindo features (${features(a.features)}) ${a.i}/${a.n}`,
    "poses.matching": (a: Args) => `Casando ${a.pairs} pares`,
    "poses.matchingBlock": (a: Args) => `Casando ${a.pairs} pares (bloco ${a.i}/${a.n})`,
    "poses.calibrating": () => "Calibrando grafo de vistas",
    "poses.globalMapper": () => "Reconstruindo (mapeador global)",
    "poses.retry": (a: Args) => `2ª tentativa · ${a.step}`,
    "poses.incremental": () => "Poucas câmeras registradas; tentando mapeador incremental",
    "poses.registering": (a: Args) => `Registrando imagem #${a.image}`,
    "poses.flattening": () => "Forçando órbita plana",
    "poses.ready": (a: Args) => `Poses prontas · ${features(a.features)}`,
    "poses.webFeatures": () => "Extraindo features",
    "poses.webFeaturesN": (a: Args) => `Features ${a.i}/${a.n}`,
    "poses.webMatching": (a: Args) => `Casando ${a.i}/${a.n} pares`,
    "poses.webRegistered": (a: Args) => `Registradas ${a.i}/${a.n} câmeras`,

    "train.alphaDataset": () => "Montando frames com fundo transparente",
    "train.loading": () => "Carregando dataset na GPU",
    "train.loadingNamed": (a: Args) => `Carregando ${a.name} na GPU`,
    "train.step": (a: Args) => `Passo ${a.iter}/${a.total} · ${fmtCount(Number(a.splats))} splats`,
    "train.done": () => "Treino concluído",

    "export.readingPly": () => "Lendo o .ply",
    "export.merging": () => "Fundindo gaussianas parecidas",
    "export.writingPly": () => "Gravando o .ply",
    "export.sorting": () => "Ordenando (Morton)",
    "export.quantizing": () => "Quantizando escalas e cores",
    "export.clustering": () => "Agrupando as bandas SH (k-means)",
    "export.textures": () => "Montando as texturas",
    "export.compressing": () => "Comprimindo as texturas (WebP)",
    "export.writingSog": () => "Gravando o .sog",
    "export.done": () => "Pronto",
  },
};

export type Messages = typeof ptBR;
