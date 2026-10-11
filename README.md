# Gausify Orbit

App desktop (Tauri 2 + React) que transforma **vídeos orbitais** — inclusive
gerados por IA — em **Gaussian Splatting**, 100% local, no macOS (Apple
Silicon) e no Windows (com aceleração CUDA no COLMAP quando há GPU NVIDIA).

```
vídeo ──► Frames ─────────► Recorte ─────────► Poses ──────────────► Treino ─────────► splat.ply
          ffmpeg            (opcional)          COLMAP 4              Brush
          decodificação     Vision (macOS),     SIFT reforçado,       wgpu (Metal,
          por hardware +    BiRefNet+DirectML   matching circular,    Vulkan/DX12),
          nitidez (Rust)    (Windows)           mapeador global       prévias ao vivo
```

## Instalação

### macOS

```bash
brew install ffmpeg colmap
pnpm install
pnpm setup:tools        # baixa o Brush (sidecar do Tauri) e confere ffmpeg/COLMAP
```

### Windows (usuário final)

1. Baixe e execute o `Gausify Orbit_<versão>_x64-setup.exe` (gerado pelo
   GitHub Actions, ver abaixo). Ele instala só para o usuário atual, sem
   pedir administrador, e baixa o WebView2 se faltar.
2. Na primeira execução o app mostra **"Instalar ferramentas"**: baixa o
   COLMAP (com CUDA se houver GPU NVIDIA) e o ffmpeg, confere o SHA-256 e
   instala em `%LOCALAPPDATA%\SplatForge\tools`. O Brush já vem no
   instalador.

Sem assinatura de código, o Windows SmartScreen mostra "O Windows protegeu
o computador": clique em **Mais informações → Executar assim mesmo**. Para
eliminar o aviso é preciso assinar o instalador (certificado de code signing
ou Azure Trusted Signing).

### Windows (desenvolvimento)

Pré-requisitos: Node 20+, pnpm, Rust com o toolchain MSVC (Visual Studio
Build Tools, carga "Desenvolvimento para desktop com C++") e WebView2 (já vem
no Windows 11). Para CUDA, um driver NVIDIA recente.

```powershell
pnpm install
pnpm setup:tools        # Brush + COLMAP 4.2.1 + ffmpeg, com SHA-256 conferido
```

O `setup:tools` escolhe o COLMAP com CUDA se encontrar `nvidia-smi` (use
`--cpu` para forçar a versão sem CUDA) e instala COLMAP e ffmpeg em
`%LOCALAPPDATA%\SplatForge\tools`, onde o app procura. Também são
encontrados executáveis instalados por winget, scoop, Chocolatey ou no PATH.

Com CUDA, extração SIFT, matching e bundle adjustment do COLMAP rodam na GPU.
O SIFT reforçado (forma afim + agrupamento de escalas) só existe na CPU; o
COLMAP faz esse fallback sozinho e o matching continua na GPU.

## Uso

```bash
pnpm tauri dev          # desenvolvimento
pnpm tauri build        # macOS: .app e .dmg · Windows: instalador NSIS
```

### Gerar os instaladores no GitHub

O workflow `.github/workflows/release.yml` compila em máquinas Windows e
macOS do GitHub e publica os instaladores numa release de rascunho:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Ou dispare manualmente em **Actions → release → Run workflow**.

1. Arraste um vídeo para a janela.
2. Se o objeto "derrete" no fim da órbita, marque um trecho estável com
   **Início aqui / Fim aqui**.
3. Escolha um preset e clique em **Gerar splat**.

### Recorte do sujeito

Com **"Isolar o sujeito"** ligado, cada frame ganha uma máscara do sujeito e o
Brush treina com o fundo transparente: o splat sai só com a pessoa ou objeto,
sem paredes, pessoas ao fundo ou os riscos em volta.

| Método | Sistema | Como | Medido |
|---|---|---|---|
| IA (Vision) | macOS 14+ | Recorte nativo do sistema, na GPU, via o sidecar `splatforge_mask` (compilado pelo `build.rs`) | ~40 ms/frame num M4 |
| Pessoas | Windows | MODNet (Apache 2.0) com ONNX Runtime + DirectML | 144 ms/frame na CPU de um M4; GPU não medida |
| Objetos | Windows | BiRefNet-lite (MIT) com ONNX Runtime + DirectML | ~4 s/frame numa RTX 4070 Ti Super |
| RMBG | Windows, macOS (Apple Silicon) | RMBG-1.4 (BRIA AI, **não comercial**) com ONNX Runtime + DirectML / CoreML na GPU | ~0,4 s/frame num M4 (+~5 s para compilar no CoreML) |
| Cor sólida | todos | Chroma key em Rust: distância RGB à cor de fundo (detectada pela mediana das bordas ou escolhida; áreas mais iluminadas da tela também contam), tolerância, suavização e preenchimento a partir das bordas | 120 frames em 0,09 s |

O BiRefNet é lento em qualquer GPU porque as convoluções deformáveis viram
~16 mil operações minúsculas no ONNX; o MODNet é uma rede convolucional
simples e, nos frames testados (pessoas de frente e de costas), recortou tão
bem quanto ele. O RMBG-1.4 tem bordas mais limpas que o Vision (sem o halo
claro em volta do cabelo), mas a licença é não comercial: por isso ele só é
baixado (176 MB, por **"Instalar ferramentas"**) quando escolhido. No macOS
roda no CoreML só com CPU e GPU: compilar para o Neural Engine pode travar o
`ANECompilerService` do sistema.

No Windows os modelos (MODNet 26 MB, BiRefNet 224 MB) são baixados por
**"Instalar ferramentas"** e o `DirectML.dll` 1.15.4 vai dentro do instalador.

**Despill** (qualquer método): com fundo verde ou azul, a luz rebatida pela
tela tinge o sujeito inteiro, não só a borda. A cor do fundo é a mediana dos
pixels que a máscara marca como fundo; em todo o sujeito o canal dessa cor é
limitado ao maior dos outros dois e a luminância tirada volta como cinza (o
sujeito não escurece; amarelos e cianos ficam intactos). Fundos pouco
saturados ou de outra cor são ignorados. Numa tela verde com sujeito de cor
parecida, **RMBG + despill** dá o melhor resultado: a IA faz o recorte e o
despill tira o verde.

Opcionalmente as máscaras também vão para o COLMAP ("Usar o recorte também
nas poses"): ajuda quando o fundo muda entre frames, mas tira pontos de apoio
de sujeitos com pouca textura, por isso vem desligado.

Cada vídeo ganha uma pasta de projeto (`<video>_splat/` por padrão). Para
mudar só os parâmetros de treino, escolha **Executar a partir de → Treino**.
Assim frames e poses são reaproveitados.

```
<video>_splat/
├── project.json        estado (frames, métricas, resultado)
├── dataset/            entrada do Brush: images/ + sparse/0/
├── masks/              recortes (cinza, 255 = sujeito)
├── dataset_alpha/      com recorte: PNGs RGBA + images.bin com nomes .png
├── colmap/             database.db, pairs.txt, modelos brutos
├── training/           snapshots intermediários (mantém os 2 últimos)
├── splat.ply           resultado final
└── export/             exportações otimizadas (.sog / .ply)
```

### Exportação otimizada

Com o splat treinado, o painel **Exportação** gera uma versão menor sem
retreinar (dá para testar ajustes e alternar **Ver no 3D / Ver original**):

| Ajuste | O que faz |
|---|---|
| **SOG comprimido** | Formato do PlayCanvas (v2): texturas WebP sem perdas com posições em 16 bits, quatérnios "smallest three", escalas e cores em codebooks de 256 e as bandas SH numa paleta k-means de até 64k vetores. Abre no Spark (o visualizador do app), PlayCanvas e SuperSplat. |
| **Reduzir** | Funde gaussianas parecidas (vizinhas, de forma e cor próximas) até remover a porcentagem pedida. A opacidade conserva a "tinta" (opacidade × área), então as superfícies não abrem buracos. |
| **Opacidade mín.** | Descarta splats quase invisíveis. |
| **Bandas SH** | Corta o brilho dependente do ângulo; cada grau a menos encolhe bastante o .ply. |

Port do que o WebGaussian faz no navegador, em Rust com rayon. Medido num M4:

| Entrada | Saída | Tempo |
|---|---|---|
| 111k splats, SH 1, 10,3 MB | `.sog` 2,1 MB (4,9×) | 2,5 s |
| idem, −50%, opac. ≥ 2%, SH 0 | `.sog` 0,69 MB (14,9×) | 0,2 s |
| 1M splats, SH 3, 236 MB | `.sog` 18 MB (12,8×) | 15,7 s |
| idem, −50% | `.ply` 118 MB | 1,2 s |

O `.sog` foi conferido com o leitor de referência (`splat-transform` da
PlayCanvas): erro de posição de 5×10⁻⁵ da extensão da cena, rotação abaixo de
1,2° (p99) e coeficientes SH com erro mediano de 0,001. O WebP sai do libwebp
(compilado do fonte pelo `libwebp-sys`), ~18% menor que o encoder em Rust puro.

### Linha de comando (lote)

O mesmo pipeline roda sem interface, com eventos em JSON por linha:

```bash
cd src-tauri
cargo run --release --example cli -- config.json      # formato de JobConfig
cargo run --release --example cli -- scene <projeto>  # enquadramento da cena
cargo run --release --example cli -- export <projeto> [opcoes.json]
```

`opcoes.json` segue o `ExportOptions`, por exemplo
`{"format": "sog", "decimate": 0.5, "minOpacity": 0.02, "shDegree": 0}`.

## Por que esse pipeline

| Escolha | Motivo |
|---|---|
| Decodificação com VideoToolbox e seleção por nitidez | Extrai N×K candidatos e fica com o mais nítido de cada janela, uniformemente distribuídos na órbita. O cálculo roda em Rust com rayon, em paralelo. |
| Câmera única pinhole | Vídeo gerado não tem distorção de lente; menos parâmetros deixam a calibração mais estável. |
| Matching circular | Cada frame é casado com seus vizinhos, com wrap-around para fechar a volta, mais saltos longos (16/32/64). Sai muito mais barato que o matching exaustivo e não depende de vocab tree. |
| `view_graph_calibrator` + `global_mapper` | O mapeador global do COLMAP 4 (ex-GLOMAP) é bem mais rápido que o incremental. O incremental só entra como fallback, quando o global registra menos de 60% das câmeras. |
| SIFT reforçado | Limiar de detecção baixo (0,002), agrupamento de escalas e forma afim, com 20 vizinhos casados. Num vídeo real de IA com a volta por trás de uma cabeça (cabelo liso, fundo desfocado), o SIFT padrão registrou 83/107 câmeras; o reforçado registrou 107/107 em ~110 s. Com o SIFT padrão, se ficar abaixo de 95% das câmeras, o app repete automaticamente com o reforçado. |
| ALIKED + LightGlue (opcional) | Também registrou 107/107 nesse vídeo, mas levou 16,5 min na CPU e teve erro de reprojeção maior (1,5 px contra 0,84 px). |
| Brush como sidecar | Treino na GPU via Metal, sem CUDA. Exporta snapshots que o visualizador (Spark/three.js) recarrega ao vivo; o visualizador só renderiza quando algo muda, para não disputar a GPU com o treino. |
| Perda SSIM desligada por padrão | No Metal ela triplica o custo de cada passo (8,8 → 26,7 passos/s sem ela). Num vídeo real (107 frames, 960×1280), sem SSIM 15k passos deram 32,6 dB em 8,6 min; com SSIM, 7k passos deram 31,8 dB em 9,3 min e 20k deram 33,2 dB em ~28 min. O preset Qualidade mantém SSIM. |
| Trava de ângulo no visualizador | A órbita da câmera fica na faixa de alturas que o vídeo filmou (+20°), onde há dados. |
| PSNR em frames separados | Mede a consistência 3D do vídeo: abaixo de ~25 dB, o vídeo de IA provavelmente não é geometricamente coerente. |

Medição num vídeo sintético de teste (960×540, 120 frames, M4 16 GB):
frames em 0,6 s; poses em 25 s (120/120 câmeras, 0,57 px); 3 mil passos
de treino em 94 s, chegando a PSNR 35,7 dB.

## Desenvolvimento da interface sem o backend

`pnpm dev:mock` sobe só a interface no navegador e reproduz eventos gravados
pela CLI. Configure em `.env.local`:

```
VITE_MOCK_VIDEO=/caminho/video.mp4
VITE_MOCK_EVENTS=/caminho/eventos.jsonl   # stdout da CLI
VITE_MOCK_SCENE=/caminho/scene.json       # saída de `cli scene`
```

Na URL, `?os=windows` simula a interface do Windows e `?events=…&scene=…`
troca os arquivos sem editar o `.env.local`. `?project=<pasta>` abre um projeto
real; a exportação é simulada e devolve o `lastExport` gravado nele por
`cli export`. Use uma pasta fora de `~/Downloads`/`~/Documents`: o macOS pode
bloquear o acesso do Vite a elas e o servidor trava.

## Versão web (tudo no navegador)

`web/` roda o pipeline inteiro no navegador, com a **mesma interface** do app
desktop: o `web/src/main.tsx` monta o `src/App.tsx` e os componentes de
`src/components/`. O que muda entre as duas versões fica atrás da interface
`Host` (`src/lib/api.ts`):

- **Desktop** (`src/lib/tauri.ts`): comandos do backend em Rust via Tauri,
  diálogos e "Mostrar no Finder".
- **Web** (`web/src/host/`): os mesmos comandos (`startJob`, `loadProject`,
  `exportSplat`…) e os mesmos eventos (`JobEvent`), implementados com WebGPU,
  WASM e OPFS. `job.ts` segue as regras do `pipeline.rs` (etapas, retomada,
  `project.json`).

Uma funcionalidade nova na interface aparece nas duas versões; o que depende
do sistema usa `tools.os` (`"web"` no navegador), como já era feito para
macOS e Windows.

```bash
pnpm web:wasm           # clona o Brush num commit fixo, aplica web/brush/*.patch e compila (~5 min)
pnpm web:dev            # http://localhost:1440
```

O `web:wasm` precisa de `rustup` e `wasm-pack`; instala sozinho um toolchain
Rust 1.99 lado a lado (o Burn do Brush exige ≥ 1.95) sem mexer no `stable`.
Navegador: Chrome ou Edge desktop com WebGPU e a extensão `subgroups`.

Diferenças em relação ao desktop:

- **Projetos** ficam no armazenamento do navegador (OPFS), em
  `/projects/<nome do vídeo>`, com o mesmo layout e o mesmo `project.json`;
  ficam os 3 mais recentes. Ao soltar o mesmo vídeo de novo, o projeto volta
  com as etapas prontas. **Importar…** (no lugar de "Alterar…") copia uma
  pasta de projeto do desktop ou um dataset COLMAP (`images/` + `sparse/0`).
- **Arquivos:** "Mostrar no Finder" vira **Baixar**.
- **Recorte:** RMBG-1.4 e cor sólida. O RMBG (176 MB, licença não comercial)
  é baixado pelo mesmo aviso de primeiro uso do desktop, com SHA-256
  conferido.
- **Poses:** SfM próprio (abaixo); as opções do COLMAP (features, modelo de
  câmera, órbita plana) não aparecem.
- **Exportação:** a mesma do desktop (PLY ou SOG, poda, corte de bandas SH e
  decimação), portada para TypeScript em `web/src/export/` e rodando num
  worker. O WebP sem perdas vem do libwebp em WASM do splat-transform (MIT,
  `web/src/export/webp/`), porque o canvas pré-multiplica o alfa.
- **fps do vídeo:** lido do contêiner MP4/MOV, como o ffprobe; em outros
  formatos é medido durante a reprodução.

O que foi preciso consertar para o treino rodar (e rodar rápido) no navegador:

- **Commit do Brush:** fixado antes do #546, que leva o kernel
  `project_forward` a 12 storage buffers; o Chrome expõe só 10 por estágio
  no Metal e o treino não roda no Mac.
- **`enable subgroups;`:** o writer WGSL do naga (wgpu 30) usa as funções de
  subgroup sem declarar a extensão, e o Chrome recusa os shaders. O
  `web/src/brush.ts` acrescenta a diretiva antes de entregar o WGSL ao
  navegador.
- **Patch `web/brush/brush-js.patch`:**
  - Construtores estáticos rodam uma vez só. Sem isso, o wasm-ld trata o
    módulo como "command" e embrulha **toda** export (até `malloc`) para
    rodar de novo os milhares de registros `inventory` do CubeCL a cada
    chamada JS→WASM. O treino ficava ~25× mais lento (4 passos/s).
  - Autotune do CubeCL em `Full`: nos outros níveis o Burn mede a vazão de
    pico da GPU com um `sync` bloqueante, que dá panic no WASM no primeiro
    reduce.
  - `trainSteps` não consulta de novo um stream já encerrado (panic "async fn
    resumed after completion" no fim do treino).
  - `Training.exportPly()`: o mesmo export do Brush nativo (piso de escala
    embutido, unidades do dataset restauradas).

### Etapas no navegador

1. **Frames:** o navegador decodifica o vídeo (um seek por candidato).
   Candidatos uniformes no trecho (`frames × candidatos`, no máximo o fps do
   vídeo), escala que cabe no lado máximo sem ampliar, e o mais nítido de cada
   janela (variância do Laplaciano na miniatura de 640 px, luma Rec.709), como
   no `frames.rs`. O seek vai meio quadro adiante para pegar o quadro mais
   próximo do instante, como o filtro `fps` do ffmpeg.
2. **Recorte:**
   - **RMBG-1.4**: ONNX Runtime Web com WebGPU, com o mesmo pré/pós-
     processamento do `masks.rs`, modelo do mesmo endereço fixado do
     instalador. É o único modelo de IA da versão web: o Vision é só do macOS,
     o BiRefNet é lento até em GPU dedicada e o MODNet (pessoas) ficou de fora
     por decisão do projeto.
   - **Cor sólida** e **despill**: port exato do Rust (testes em
     `web/src/pipeline/*.test.ts`, `pnpm test:web`, com os casos do
     `masks.rs`).
   - `dataset_alpha`: PNG RGBA com alfa direto (encoder próprio: o canvas
     pré-multiplica o alfa) e o `images.bin` com os nomes `.png`. "Usar o
     recorte também nas poses" descarta as features onde a máscara é 0.
3. **Poses:** SfM sem o COLMAP (`web/src/sfm/`):
   1. **Features:** AKAZE do OpenCV.js (`@techstark/opencv-js` 4.12, que não
      traz SIFT nem as funções de geometria epipolar) em Web Workers.
   2. **Matching:** força bruta por distância de Hamming num compute shader
      WebGPU, com teste de razão e checagem mútua; mesmos pares do app (janela do
      preset + saltos 16/32/64, órbita fechada).
   3. **Verificação:** matriz fundamental por RANSAC (8 pontos normalizado).
   4. **Reconstrução incremental** num worker (`web/src/sfm/`): tracks por
      union-find, par inicial escolhido por paralaxe (≥ 2°) e número de pontos,
      com até 3 tentativas em regiões diferentes da órbita se a primeira
      registrar menos de 90% das câmeras, PnP partindo da pose do frame
      vizinho (com DLT + RANSAC de reserva), triangulação, BA local a cada frame
      e global periódico (Levenberg–Marquardt com complemento de Schur, Huber,
      focal compartilhada). No fim, BA com limiar decrescente reaproveita as
      observações que fecham a órbita.
4. **Treino:** Brush em WASM + WebGPU, com prévias a cada "Prévia a cada"
   passos no visualizador.

Medido num vídeo de tela verde sintético (`synth-orbit.py --green`: 120
quadros 960×720, 24 fps, verde rebatido nos objetos), preset Prévia (90
frames), no Chrome headless do M4:

| Etapa | Tempo | Resultado |
|---|---|---|
| Frames | 3,1 s | 90 de 120 candidatos |
| Recorte, cor sólida | 2,4 s | IoU 0,91 com as máscaras verdadeiras (mín. 0,83) |
| Recorte, RMBG-1.4 (WebGPU) | 32 s (+12 s no 1º uso: download e verificação) | IoU 0,996 (mín. 0,99), ~0,36 s/frame |
| Poses | 29–31 s | 90/90 câmeras, posição 0,15% do raio, rotação 0,12°, focal −0,4% |
| PNG com transparência | 4,5 s | pixels do sujeito com verde dominante: 16% → 0% |
| Treino (3 mil passos, cor sólida) | — | PSNR 36,7 dB, 0,4% das gaussianas visíveis verdes |

O chroma key não chega a IoU 1 porque o cubo tem manchas verde-azuladas que
encostam na borda e saem junto com a tela (o mesmo acontece no app desktop);
o RMBG + despill as mantém. Num painel ou aba oculta o navegador limita os
seeks e a extração de frames fica ~10× mais lenta.

Poses medidas num vídeo orbital sintético de 120 frames 960×720 com poses
verdadeiras conhecidas (`synth-orbit.py`: textura procedural, chão e parede
ao fundo):

| | Tempo | Câmeras | Pontos | Focal | Posição (mediana) | Rotação (mediana) | PSNR do treino |
|---|---|---|---|---|---|---|---|
| Navegador | 37 s | 120/120 | 14 mil | +0,22% | 0,27% do raio | 0,085° | 39,7 dB |
| COLMAP 4.2 nativo (SIFT, global) | 14 s | 120/120 | 2 mil | +0,07% | 0,73% do raio | 0,19° | 36,2 dB |
| Poses verdadeiras | — | — | — | — | — | — | 41,2 dB |

PSNR: Brush web, 5 mil passos a 960 px, SH 1, 8 vistas de avaliação. O treino
roda a ~40 passos/s nesse dataset (contra ~3× isso no Brush nativo).

Para repetir a medição (`scripts/web-bench/`, com `WEB_FS_ANY=1 pnpm web:dev`
rodando):

```bash
python3 scripts/web-bench/synth-orbit.py /tmp/orbit 120 960 720
```
```bash
ffmpeg -framerate 24 -i /tmp/orbit/images/%05d.jpg -c:v libx264 -pix_fmt yuv420p /tmp/orbit.mp4
```
```bash
node scripts/web-bench/headless.mjs poses "http://localhost:1440/?video=/@fs/tmp/orbit.mp4&dataset=/@fs/tmp/orbit" --out /tmp/orbit-web
```
```bash
python3 scripts/web-bench/eval-poses.py /tmp/orbit/gt.json /tmp/orbit-web/sparse/0
```
```bash
node scripts/web-bench/headless.mjs train "http://localhost:1440/?video=/@fs/tmp/orbit.mp4&dataset=/@fs/tmp/orbit-web" --steps 5000 --max-res 960
```

O `headless.mjs` usa a interface num Chrome headless (sem o throttling de abas
ocultas). `?video=` abre o vídeo como se ele fosse solto na janela;
`?dataset=` faz o projeto desse vídeo começar com um dataset COLMAP (imagens e
poses), para treinar ou recalcular as poses a partir da etapa certa. Modos:
`video` (do vídeo ao splat, com `--mask chroma|rmbg`), `poses`, `train` e
`profile` (perfil de CPU do treino; com `pnpm web:wasm --profiling` sai com
os nomes das funções do WASM).

Ainda não foi testado em vídeos reais de IA (desfoque, pouca textura,
elementos que se mexem), onde o SIFT reforçado e o mapeador global do COLMAP
fazem diferença.

### Publicação na Cloudflare

A versão web é publicada como **assets estáticos no Cloudflare Workers**
(`wrangler.jsonc`), que não acordam o Worker e não são cobrados. O código de
Worker (`worker/index.ts`) só roda nos links de compartilhamento (`/api/*` e
`/s/*`). O endereço é `gausify-orbit.<subdomínio>.workers.dev`; para um
domínio próprio, veja o comentário de `routes` no `wrangler.jsonc`.

- **Automático:** o `.github/workflows/web.yml` verifica cada PR (tipos,
  testes, build e limite de tamanho) e publica a cada push na `main`. Ele
  precisa do segredo `CLOUDFLARE_API_TOKEN` no repositório (token com o modelo
  "Edit Cloudflare Workers"). O Brush em WASM é compilado no CI e fica em
  cache até o commit fixo ou os patches mudarem.
- **Manual:** `pnpm web:deploy` (build + `wrangler deploy`, com a conta do
  `wrangler login`).

Detalhes que o Workers impõe:

- **25 MiB por arquivo:** o `.wasm` do ONNX Runtime (~27 MB) sai gzipado
  (~7 MB) e o app descompacta no navegador (`web/ort-wasm-plugin.ts`).
- **`web/public/_headers`:** cache imutável em `/assets`, `no-cache` no
  `sw.js` e `Referrer-Policy: same-origin`. O Hugging Face responde 404 sem
  CORS quando o Referer é um `*.workers.dev`, e o download do RMBG falharia;
  o `segment.ts` também baixa o modelo sem Referer.
- **Offline:** o service worker (`web/sw-plugin.ts`) guarda o build inteiro
  na primeira visita e avisa quando há versão nova. Ele não intercepta `/s/*`
  nem `/api/*`.

### Compartilhar por link

Depois de exportar, **Compartilhar** (no bloco de exportação, desktop e web)
sobe o `.sog`/`.ply` para o R2 da Cloudflare e gera um link privado
`/s/<id>`, com QR code. Quem tem o link abre o splat num visualizador próprio
(`web/share.html`, PlayCanvas): arrastar/pinça no desktop e no celular, e os
botões **VR** (óculos com WebXR, como o Quest) e **AR** (celulares Android)
quando o aparelho tem suporte. A câmera e a sessão XR são portadas do player
do WebGaussian: no XR a cena vira uma maquete de ~1,4 m na frente do usuário,
com analógico para andar e snap turn. O link leva o enquadramento do vídeo
(centro, "up", primeiro frame, lente e faixa de altura filmada), então abre
igual ao **Recentrar** do app.

No XR, diferente do WebGaussian (um objeto isolado), a cena orbital tem fundo
e splats grandes e soltos perto das câmeras. Na maquete eles envolviam a
cabeça: no Quest 3 isso derrubava o fps e lavava as cores com uma névoa
branca. Por isso, só durante a sessão, o visualizador:

- descarta no shader (`setWorkBufferModifier`) os splats fora de uma esfera de
  0,8 × o raio da órbita das câmeras. No projeto de teste isso tira 24% dos
  splats, mas são os maiores (14× a escala média dos que ficam);
- gira a maquete para o lado filmado (o primeiro frame) ficar de frente;
- liga a ordenação radial do PlayCanvas (menos artefatos ao virar a cabeça) e
  a foveação fixa (0,5).

Para comparar no óculos, a URL do link aceita `?crop=` (0 desliga o recorte),
`?foveation=`, `?xrScale=` (resolução do XR) e `?sort=linear`. `?xrPreview=1`
monta a mesma maquete no navegador comum, com a câmera na altura da cabeça.

O visualizador usa WebGL 2, como o WebGaussian. `?gpu=webgpu` troca para
WebGPU, onde o PlayCanvas ordena os splats na GPU (`?gpuSort=0` volta a
ordenar na CPU), e um chip mostra o backend em uso. No VR isso depende da API
`XRGPUBinding` (WebXR/WebGPU Binding), ainda experimental: no Quest Browser
146 ou mais novo, ligue "WebXR experimental features" em `chrome://flags`.
Sem ela, num aparelho com VR/AR, o visualizador fica no WebGL 2 para os botões
não sumirem, e o chip avisa. Cuidado: num Chrome sem o binding,
`requestAdapter({ xrCompatible: true })` nunca responde e trava o WebGPU da
aba, e o PlayCanvas pede isso sozinho quando existe `navigator.xr`. Por isso o
visualizador só passa `xrCompatible` ao WebGPU quando o binding existe.

- **Privado:** o id tem ~71 bits aleatórios e a página não é indexada nem
  manda Referer. Quem tem o link pode ver (e baixar) o splat.
- **Validade:** `SHARE_TTL_HOURS` no `wrangler.jsonc` (padrão 6 h). O Worker
  trata o vencido como inexistente na hora, um cron a cada 30 min apaga do
  R2, e quem criou pode **Revogar** antes (o app guarda o token de remoção no
  aparelho).
- **Limites:** `SHARE_MAX_MB` (95 MiB; o Cloudflare recusa corpos acima de
  100 MB), só arquivos que começam como `.sog` (ZIP) ou `.ply`, e 10 envios
  por minuto por IP (`ratelimits`).
- **Desktop:** envia para a versão publicada; o Worker libera CORS para a
  webview do Tauri. `VITE_SHARE_ORIGIN` troca o servidor (desktop e
  `web:dev`).

Configuração, uma vez só (o deploy falha sem o bucket):

```bash
pnpm exec wrangler r2 bucket create gausify-orbit-shares
pnpm exec wrangler r2 bucket lifecycle add gausify-orbit-shares expira-links s/ --expire-days 1
```

A regra de ciclo de vida é só a rede de segurança caso o cron pare (o R2 não
expira em menos de um dia). O token do CI (`CLOUDFLARE_API_TOKEN`, modelo
"Edit Cloudflare Workers") já inclui a permissão de R2.

Para testar localmente: `pnpm web:build`, `pnpm exec wrangler dev` (R2 e
limite de taxa simulados em `.wrangler/`) e abrir `http://localhost:8787`.
Para testar a interface do desktop contra ele: `VITE_SHARE_ORIGIN=http://localhost:8787 pnpm dev:mock`
e `wrangler dev --var SHARE_ORIGINS:http://localhost:1430`.

## Testes

```bash
pnpm test:rust
```
